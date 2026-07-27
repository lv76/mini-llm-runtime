import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                denom = v_hat.sqrt().add(eps)
                update = m_hat / denom + wd * p.data

                p.data.add_(update, alpha=-lr)

        return loss
