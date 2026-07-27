from typing import Any

import torch
import math

class MyFlashAttnAutogradFunctionClass_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        d = q.shape[-1]
        Scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        if is_causal is True:
            mask = torch.tril(
                torch.ones(1, 1, q.shape[-2], q.shape[-2])
                ).bool()
            Scores = Scores.masked_fill(~mask, float("-inf"))

        p = torch.softmax(Scores, dim=-1)
        out = p @ v

        lse = torch.logsumexp(Scores, dim=-1)

        ctx.save_for_backward(q, k, v, lse)
        return out
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, lse = ctx.saved_tensors

        d = q.shape[-1]
        scale = 1 / math.sqrt(d)
        scores = q @ k.transpose(-2, -1) * scale
        p = torch.exp(scores - lse.unsqueeze(-1))

        dv = p.transpose(-2,-1) @ do

        dp = do @ v.transpose(-2,-1)

        ds = p * (dp - (p*dp).sum(dim = -1, keepdim = True))

        dq = ds @ k * scale
        dk = ds.transpose(-2,-1) @ q * scale

        return dq, dk, dv, None
    
