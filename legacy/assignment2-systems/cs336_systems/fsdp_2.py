from __future__ import annotations
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear

class FullSeparateDistributedParallel(nn.Module):
    def __init__(self, module, compute_dtype) -> None:
        super().__init__() 
        self.module = module
        self.compute_dtype = compute_dtype

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.sharded_param_infos: list[dict] = []

        self.replicated_para_infos: list[dict] = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _shared_parameter(self):

        sharded_param_ids: set[int] = set()
        for module_name, submodule in self.named_modules():

            if not isinstance(submodule, (Embedding, Linear)):
                continue
        
            parameter = submodule.weight

            if id(parameter) in sharded_param_ids:
                continue

            parameter_name = (f"{module_name}.weight")

            original_shape = parameter.shape
            original_numel = parameter.numel()

            shared_numel = torch.ceil(original_numel, self.world_size)

            padded_numel = shared_numel * self.world_size
            padding_numel = padded_numel - original_numel

            full_flat_parameter = (parameter.detach().reshape(-1).contiguous)

            if padding_numel > 0:
                full_flat_parameter = torch.cat([full_flat_parameter, full_flat_parameter_zero(padding_numel)],dim=0)
            
            start = self.rank * shared_numel
            end = (self.rank + 1) * shared_numel 

            local_shared = (full_flat_parameter[start:end].clone().contiguous())

            parameter.data = local_shared

            param_info = {
                "name": parameter_name,
                "module": submodule,
                "parameter": parameter,
                "original_shape": original_shape,
                "original_numel": original_numel,
                "shard_numel": shared_numel,
                "padded_numel": padded_numel,
                "local_shared": local_shared
            }

            self.sharded_param_infos.append(param_info)
            sharded_param_ids.add(id(parameter))

        for parameter_name, parameter in (
            self.module.named_parameters()
        ):
            if id(parameter) in sharded_param_ids:
                continue

            self.replicated_param_infos.append(
                {
                    "name": parameter_name,
                    "parameter": parameter,
                }
            )

            





            




