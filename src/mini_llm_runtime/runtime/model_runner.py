from __future__ import annotations

from dataclasses import dataclass

import torch

from .scheduler import Scheduler, SchedulerOutput
from .request import Request, RequestStatus
from mini_llm_runtime.model.qwen2 import TinyQwen2ForCausalLM
from .paged_kv_cache import PagedKVCache


@dataclass
class ModelBatch:
    """Flattened model inputs for all requests in one scheduler step."""

    request_ids: list[str] 
    #当前模型输入
    input_token_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    request_phases: list[RequestStatus]

    # PagedAttention metadata
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor

@dataclass
class GenerationStepResult:
    scheduler_output: SchedulerOutput
    model_batch: ModelBatch
    packed_logits: torch.Tensor
    sampled_token_ids: dict[str, int]
    finished_requests: list[Request]

def build_model_batch(
    scheduler_output: SchedulerOutput,
    block_size: int,
    device: torch.device | str = "cpu",
) -> ModelBatch:
    request_ids: list[str] = []
    input_token_ids: list[int] = []
    positions: list[int] = []
    query_start_loc = [0]
    context_lens: list[int] = []
    block_tables: list[list[int]] = []
    slot_mapping: list[int] = []
    request_phases: list[RequestStatus] = []

    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")

    def append_slots(request, token_positions: range | list[int]) -> None:
        for position in token_positions:
            logical_block = position // block_size
            if logical_block >= len(request.block_table):
                raise ValueError(
                    f"Request {request.request_id} has no KV cache block "
                    f"for position {position}"
                )
            physical_block = request.block_table[logical_block]
            block_offset = position % block_size
            slot_mapping.append(physical_block * block_size + block_offset)

    for request in scheduler_output.prefill_requests:
        request_phases.append(RequestStatus.PREFILLING)
        scheduled_tokens = scheduler_output.num_scheduled_tokens[request.request_id]
        start = request.num_computed_tokens
        tokens = request.prompt_token_ids[start : start + scheduled_tokens]

        if len(tokens) != scheduled_tokens:
            raise ValueError(
                f"Request {request.request_id} expected {scheduled_tokens} "
                f"prefill tokens, got {len(tokens)}"
            )

        request_ids.append(request.request_id)
        input_token_ids.extend(tokens)
        positions.extend(range(start, start + scheduled_tokens))
        query_start_loc.append(query_start_loc[-1] + scheduled_tokens)
        context_lens.append(start + scheduled_tokens)
        block_tables.append(list(request.block_table))
        append_slots(request, range(start, start + scheduled_tokens))

    for request in scheduler_output.decode_requests:
        request_phases.append(RequestStatus.DECODING)
        scheduled_tokens = scheduler_output.num_scheduled_tokens[
            request.request_id
        ]
        if scheduled_tokens != 1:
            raise ValueError(
                "Decode currently supports exactly one token per request"
            )

        pending_token_id = request.pending_token_id
        if pending_token_id is None:
            raise ValueError(
                f"Decode request {request.request_id} has no pending token"
            )

        request_ids.append(request.request_id)
        input_token_ids.append(pending_token_id)
        positions.append(request.num_computed_tokens)
        query_start_loc.append(query_start_loc[-1] + 1)
        context_lens.append(request.num_computed_tokens + scheduled_tokens)
        block_tables.append(list(request.block_table))
        append_slots(request, [request.num_computed_tokens])

    target_device = torch.device(device)
    max_num_blocks = max((len(table) for table in block_tables), default=0)
    padded_block_tables = [
        table + [-1] * (max_num_blocks - len(table))
        for table in block_tables
    ]
    block_tables_tensor = (
        torch.tensor(
            padded_block_tables, dtype=torch.long, device=target_device
        )
        if block_tables
        else torch.empty((0, 0), dtype=torch.long, device=target_device)
    )

    return ModelBatch(
        request_ids=request_ids,
        input_token_ids=torch.tensor(
            input_token_ids, dtype=torch.long, device=target_device
        ),
        positions=torch.tensor(
            positions, dtype=torch.long, device=target_device
        ),
        query_start_loc=torch.tensor(
            query_start_loc, dtype=torch.long, device=target_device
        ),
        context_lens=torch.tensor(
            context_lens, dtype=torch.long, device=target_device
        ),
        block_tables=block_tables_tensor,
        slot_mapping=torch.tensor(
            slot_mapping, dtype=torch.long, device=target_device
        ),
        request_phases=request_phases,
    )


def execute_model_batch(
    model: TinyQwen2ForCausalLM,
    batch: ModelBatch,
    kv_cache: PagedKVCache,
) -> torch.Tensor:
    if len(batch.request_phases) != len(batch.request_ids):
        raise ValueError(
            "request_phases length must match request_ids"
        )
    if not batch.request_ids:
        raise ValueError("Cannot execute an empty ModelBatch")
    if batch.input_token_ids.ndim != 1:
        raise ValueError("input_token_ids must be one-dimensional")
    if batch.positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if batch.slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be one-dimensional")
    num_packed_tokens = batch.input_token_ids.numel()
    if batch.positions.numel() != num_packed_tokens:
        raise ValueError(
            "positions length must match input_token_ids"
        )
    if batch.slot_mapping.numel() != num_packed_tokens:
        raise ValueError(
            "slot_mapping length must match input_token_ids"
        )
    if batch.context_lens.ndim != 1:
        raise ValueError("context_lens must be one-dimensional")
    if batch.context_lens.numel() != len(batch.request_ids):
        raise ValueError("context_lens length must match request_ids")
    if batch.block_tables.ndim != 2:
        raise ValueError("block_tables must be two-dimensional")
    if batch.block_tables.shape[0] != len(batch.request_ids):
        raise ValueError("block_tables rows must match request_ids")
    index_tensors = {
        "input_token_ids": batch.input_token_ids,
        "positions": batch.positions,
        "query_start_loc": batch.query_start_loc,
        "context_lens": batch.context_lens,
        "block_tables": batch.block_tables,
        "slot_mapping": batch.slot_mapping,
    }
    for tensor_name, tensor in index_tensors.items():
        if tensor.dtype is not torch.long:
            raise ValueError(f"{tensor_name} must use torch.long dtype")

    batch_device = batch.input_token_ids.device
    for tensor_name, tensor in index_tensors.items():
        if tensor.device != batch_device:
            raise ValueError(
                f"{tensor_name} must be on the same device as "
                "input_token_ids"
            )

    model_parameter = next(model.parameters())
    if batch_device != model_parameter.device:
        raise ValueError("ModelBatch and model must be on the same device")
    if kv_cache.device != model_parameter.device:
        raise ValueError("KV cache and model must be on the same device")
    if kv_cache.dtype != model_parameter.dtype:
        raise ValueError("KV cache dtype must match model dtype")
    if batch.query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be one-dimensional")
    if batch.query_start_loc.numel() != len(batch.request_ids) + 1:
        raise ValueError(
            "query_start_loc must contain one more entry than request_ids"
        )
    if int(batch.query_start_loc[0].item()) != 0:
        raise ValueError("query_start_loc must start at 0")
    if torch.any(batch.query_start_loc[1:] <= batch.query_start_loc[:-1]):
        raise ValueError("query_start_loc must be strictly increasing")
    if int(batch.query_start_loc[-1].item()) != num_packed_tokens:
        raise ValueError(
            "query_start_loc must end at the packed token count"
        )

    for request_index, request_id in enumerate(batch.request_ids):
        context_len = int(batch.context_lens[request_index].item())
        if context_len <= 0:
            raise ValueError(
                f"Request {request_id} context_len must be greater than 0"
            )
        num_required_blocks = (
            context_len + kv_cache.block_size - 1
        ) // kv_cache.block_size
        if batch.block_tables.shape[1] < num_required_blocks:
            raise ValueError(
                f"Request {request_id} block_table has insufficient blocks"
            )
        valid_blocks = batch.block_tables[
            request_index, :num_required_blocks
        ]
        if torch.any(valid_blocks < 0) or torch.any(
            valid_blocks >= kv_cache.num_blocks
        ):
            raise ValueError(
                f"Request {request_id} block_table contains an invalid "
                "physical block"
            )

    decode_indices = []
    for i, phase in enumerate(batch.request_phases):
        if phase is RequestStatus.DECODING:
            if int(batch.query_start_loc[i + 1] - batch.query_start_loc[i]) != 1:
                raise ValueError("Decode requires exactly one token per request")
            decode_indices.append(i)
        elif phase is not RequestStatus.PREFILLING:
            raise ValueError(
                f"Unsupported request phase {phase} for request {batch.request_ids[i]}"
            )

    # Preserve the original packed token order, including interleaved phases.
    packed_logits = model.lm_head.weight.new_empty(
        (num_packed_tokens, model.lm_head.out_features)
    )
    for i, phase in enumerate(batch.request_phases):
        if phase is not RequestStatus.PREFILLING:
            continue
        start = int(batch.query_start_loc[i])
        end = int(batch.query_start_loc[i + 1])
        request_logits = model.forward_prefill_with_cache(
            input_ids=batch.input_token_ids[start:end].unsqueeze(0),
            position_ids=batch.positions[start:end].unsqueeze(0),
            kv_cache=kv_cache,
            slot_mapping=batch.slot_mapping[start:end],
        )
        packed_logits[start:end] = request_logits.squeeze(0)

    if decode_indices:
        request_rows = torch.tensor(decode_indices, device=batch_device, dtype=torch.long)
        token_rows = batch.query_start_loc[request_rows]
        decode_logits = model.forward_decode_batch_with_cache(
            input_ids=batch.input_token_ids[token_rows].unsqueeze(1),
            position_ids=batch.positions[token_rows].unsqueeze(1),
            kv_cache=kv_cache,
            slot_mapping=batch.slot_mapping[token_rows],
            block_tables=batch.block_tables[request_rows],
            context_lens=batch.context_lens[request_rows],
        )
        packed_logits.index_copy_(0, token_rows, decode_logits.squeeze(1))

    return packed_logits


def greedy_sample(
    packed_logits: torch.Tensor,
    batch: ModelBatch,
) -> dict[str, int]:
    """Select one token from each request's final scheduled position."""
    if packed_logits.ndim != 2:
        raise ValueError(
            "packed_logits must have shape [total_tokens, vocab_size]"
        )
    if packed_logits.shape[0] != batch.input_token_ids.shape[0]:
        raise ValueError(
            "packed_logits token dimension must match ModelBatch tokens"
        )
    if len(batch.request_ids) + 1 != batch.query_start_loc.numel():
        raise ValueError(
            "query_start_loc must contain one more entry than request_ids"
        )

    sampled_token_ids: dict[str, int] = {}
    for request_index, request_id in enumerate(batch.request_ids):
        start = int(batch.query_start_loc[request_index].item())
        end = int(batch.query_start_loc[request_index + 1].item())
        if end <= start:
            raise ValueError(
                f"Request {request_id} must contain at least one query token"
            )

        last_logits = packed_logits[end - 1]
        sampled_token_ids[request_id] = int(
            torch.argmax(last_logits, dim=-1).item()
        )

    return sampled_token_ids


def run_greedy_generation(
    model: TinyQwen2ForCausalLM,
    scheduler: Scheduler,
    eos_token_id: int | None = None,
) -> list[Request]:
    """Run the reference schedule-model-sample-update loop to completion."""
    finished_requests: list[Request] = []

    while scheduler.waiting or scheduler.running:
        step_result = run_generation_step(model, scheduler, eos_token_id=eos_token_id)
        finished_requests.extend(step_result.finished_requests)

    return finished_requests


def run_generation_step(
    model: TinyQwen2ForCausalLM,
    scheduler: Scheduler,
    eos_token_id: int | None = None,
) -> GenerationStepResult:
    """Run a single schedule-model-sample-update step."""
    if not scheduler.waiting and not scheduler.running:
        raise ValueError("Cannot run generation step with no requests")

    scheduler_output = scheduler.schedule()
    if scheduler_output.is_empty:
        raise RuntimeError(
            "Scheduler made no progress while unfinished requests remain"
        )

    batch = build_model_batch(
        scheduler_output,
        block_size=scheduler.kv_cache.block_size,
        device=scheduler.kv_cache.device,
    )
    packed_logits = execute_model_batch(
        model=model,
        batch=batch,
        kv_cache=scheduler.kv_cache,
    )
    sampled_token_ids = greedy_sample(packed_logits, batch)
    reached_eos_request_ids = (
      {
          request_id
          for request_id, token_id in sampled_token_ids.items()
          if token_id == eos_token_id
      }
      if eos_token_id is not None
      else set()
    )
    finished_requests = scheduler.update_after_model(
        scheduler_output=scheduler_output,
        sampled_token_ids=sampled_token_ids,
        reached_eos_request_ids=reached_eos_request_ids,
    )

    return GenerationStepResult(
        scheduler_output=scheduler_output,
        model_batch=batch,
        packed_logits=packed_logits,
        sampled_token_ids=sampled_token_ids,
        finished_requests=finished_requests,
    )

