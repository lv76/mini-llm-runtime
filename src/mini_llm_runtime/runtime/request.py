from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED = "FINISHED"


class FinishReason(Enum):
    LENGTH = "LENGTH"
    EOS = "EOS"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class Request:
    request_id: str  #请求唯一标识
    prompt_token_ids: list[int] #输入 prompt
    max_new_tokens: int #最大生成tokens数
    arrival_time: float #用于FCFS

    output_token_ids: list[int] = field(default_factory=list) #已采样的输出
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0 #已经过模型并写入 KV Cache 的 token 数
    block_table: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None #EOS、长度上限、取消等

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)
        
    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)
    
    @property
    def num_all_tokens(self) -> int:
        return self.num_output_tokens + self.prompt_len
    
    @property
    def pending_token_id(self) -> int | None:
        if not self.output_token_ids:
            return None
        return self.output_token_ids[-1]
    
    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED

    def start_prefill(self):
        if self.status is not RequestStatus.WAITING:
            raise RuntimeError(f"Cannot start prefill for request {self.request_id}"
                               f"in status {self.status}"
                                f"Expected status: {RequestStatus.WAITING}")
        self.status = RequestStatus.PREFILLING
        
    def update_after_model(
        self,
        *,
        num_new_computed_tokens: int,
        sampled_token_id: int,
        reached_eos: bool = False,
        ) -> None:
        if self.status not in (RequestStatus.PREFILLING, RequestStatus.DECODING):
            raise RuntimeError(f"Cannot update request {self.request_id} after model"
                               f"in status {self.status}"
                               f"Expected status: {RequestStatus.PREFILLING} or {RequestStatus.DECODING}")
        self.num_computed_tokens = self.num_computed_tokens + num_new_computed_tokens

        if sampled_token_id is not None:
            self.output_token_ids.append(sampled_token_id)

        if reached_eos:
            self.status = RequestStatus.FINISHED
            self.finish_reason = FinishReason.EOS
        elif len(self.output_token_ids) >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED
            self.finish_reason = FinishReason.LENGTH
        else:
            self.status = RequestStatus.DECODING


            
        