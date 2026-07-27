import torch
import torch.nn as nn
import torch.distributed as dist


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self._grad_sync_handles = []

        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
        
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(
                    self._all_reduce_gradient
                )

    def _all_reduce_gradient(self, parameter):
        handle = dist.all_reduce(
            parameter.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        self._grad_sync_handles.append((parameter, handle))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for parameter, handle in self._grad_sync_handles:
            handle.wait()
            parameter.grad.div_(self.world_size)

        self._grad_sync_handles.clear() 