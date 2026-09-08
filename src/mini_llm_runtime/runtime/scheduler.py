from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

from .request import Request, RequestStatus
from .paged_kv_cache import PagedKVCache


@dataclass
class SchedulerOutput:
    prefill_requests: list[Request] = field(default_factory=list)
    decode_requests: list[Request] = field(default_factory=list)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        scheduled_ids = set()
        # Ensure that there are no duplicate requests across prefill and decode
        for request in self.prefill_requests:
            if request.request_id in scheduled_ids:
                raise ValueError(f"Duplicate request {request.request_id} in prefill_requests")
            scheduled_ids.add(request.request_id)
        for request in self.decode_requests:
            if request.request_id in scheduled_ids:
                raise ValueError(f"Duplicate request {request.request_id} in decode_requests")
            scheduled_ids.add(request.request_id)

        token_ids_in_num_scheduled = set(self.num_scheduled_tokens.keys())
        if token_ids_in_num_scheduled != scheduled_ids:
            raise ValueError(
                "Mismatch between scheduled requests and num_scheduled_tokens keys. "
                f"Scheduled request IDs: {scheduled_ids}, "
                f"num_scheduled_tokens keys: {token_ids_in_num_scheduled}"
            )

        for req_id, count in self.num_scheduled_tokens.items():
            if count <= 0:
                raise ValueError(f"Number of scheduled tokens for request {req_id} cannot be negative: {count}")


    @property
    def scheduled_requests(self) -> list[Request]:
        return self.prefill_requests + self.decode_requests

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


class Scheduler:
    def __init__(self, max_num_batched_tokens:int, max_num_seqs:int, kv_cache:PagedKVCache | None = None):
        self.max_num_batched_tokens = max_num_batched_tokens #最多允许max_num_batched_tokens个token被调度到模型中
        self.max_num_seqs = max_num_seqs #最多允许max_num_seqs个请求处于 Running 状态
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.request_ids: set[str] = set()
        if kv_cache is None:
            kv_cache = PagedKVCache(
                num_layers=1,
                num_blocks=max_num_batched_tokens,
                block_size=max_num_batched_tokens,
                num_kv_heads=1,
                head_dim=1,
            )
        self.kv_cache = kv_cache

    def add_request(self, request: Request):
        if request.request_id in self.request_ids:
            raise ValueError(f"Request {request.request_id} is already in the scheduler.")
        if request.status is not RequestStatus.WAITING:
            raise ValueError(f"Request {request.request_id} must be in WAITING status to be added to the scheduler.")
        self.kv_cache.add_sequence(request.request_id)
        request.block_table = self.kv_cache.block_tables[request.request_id]
        self.waiting.append(request)
        request.status = RequestStatus.WAITING
        self.request_ids.add(request.request_id)

    def schedule(self) -> SchedulerOutput:
        prefill_requests: list[Request] = []
        decode_requests: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        remaining_budget = self.max_num_batched_tokens
        num_seqs = len(self.running)
        #下一步实现 schedule() 的第一部分：只处理 waiting 请求的 prefill admission，暂时不处理 decode。
        for request in self.running:
            if remaining_budget <= 0:
                break
            if request.status is RequestStatus.DECODING:
                target_num_tokens = request.num_computed_tokens + 1
            if not self.kv_cache.reserve_tokens(request.request_id, target_num_tokens):
                continue
            decode_requests.append(request)
            num_scheduled_tokens[request.request_id] = 1
            remaining_budget -= 1
            if remaining_budget <= 0:
                break

        while self.waiting and remaining_budget > 0 and num_seqs < self.max_num_seqs:
            request = self.waiting[0]
            if request.status is not RequestStatus.WAITING:
                raise ValueError(f"Request {request.request_id} must be in WAITING status to be scheduled.")
            if request.prompt_len > remaining_budget:
                break
            if not self.kv_cache.reserve_tokens(request.request_id, request.prompt_len):
                break
            request = self.waiting.popleft()
            request.start_prefill()
            prefill_requests.append(request)
            num_scheduled_tokens[request.request_id] = request.prompt_len
            remaining_budget -= request.prompt_len
            num_seqs += 1
            self.running.append(request)
        
        return SchedulerOutput(
            prefill_requests = prefill_requests, 
            decode_requests = decode_requests, 
            num_scheduled_tokens = num_scheduled_tokens
        )

    def update_after_model(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: dict[str, int],
        reached_eos_request_ids: set[str] | None = None,
    ) -> list[Request]:
        if reached_eos_request_ids is None:
            reached_eos_request_ids = set()
    
        finished_requests: list[Request] = []

        for request in scheduler_output.scheduled_requests:
            request_id = request.request_id

            num_new_computed_tokens = scheduler_output.num_scheduled_tokens[request_id]
            sampled_token_id = sampled_token_ids[request_id]
            reached_eos = request_id in reached_eos_request_ids

            request.update_after_model(
                num_new_computed_tokens=num_new_computed_tokens,
                sampled_token_id=sampled_token_id,
                reached_eos=reached_eos,
            )

            if request.is_finished:
                finished_requests.append(request)
                self.running.remove(request)
                self.request_ids.remove(request_id)
                self.kv_cache.free_sequence(request_id)

        return finished_requests









    