import math

import torch
from torch import nn

from mini_llm_runtime.model.functional import embedding, linear, rmsnorm, silu, swiglu


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        d_out, d_in = self.weight.shape
        return linear(d_in, d_out, self.weight, in_features)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        vocab_size, d_model = self.weight.shape
        return embedding(vocab_size, d_model, self.weight, token_ids)


class SiLU(nn.Module):
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return silu(in_features)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return swiglu(
            self.w1.weight.shape[1],
            self.w1.weight.shape[0],
            self.w1.weight,
            self.w2.weight,
            self.w3.weight,
            in_features,
        )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return rmsnorm(
            self.weight.shape[0],
            self.eps,
            self.weight,
            in_features,
        )
