import torch
import torch.distributed as dist
import torch.nn as nn

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.all_params = list(params)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.local_params = []

        for i, param in enumerate(self.all_params):
            if i % self.world_size == self.rank:
                self.local_params.append(param)

        self.local_optimzer = optimizer_cls(
            self.local_params,
            **kwargs
        )

    def step(self):
        self.local_optimzer.step()

        with torch.no_grad():
            for i, param in enumerate(self.all_params):
                owner = i % self.world_size
                dist.broadcast(param, src=owner)

    def zero_grad(self):
        for param in self.all_params:
            param.grad = None


         
