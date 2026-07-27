import torch
import math
from torch.optim import Optimizer

# LR schedule:
# warmup: 0 -> max_lr
# cosine decay: max_lr -> min_lr from warmup_iters to cosine_cycle_iters
# after cosine_cycle_iters: keep min_lr
def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if(it < warmup_iters):
        lr = max_learning_rate * (it / warmup_iters)

    elif(warmup_iters <= it <= cosine_cycle_iters):
        process = (it - warmup_iters) /  (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * process))

    else:
        lr = min_learning_rate
    return lr


# AdamW:
# 1. m tracks exponential moving average of gradients.
# 2. v tracks exponential moving average of squared gradients.
# 3. Bias correction removes initialization bias.
# 4. Weight decay is decoupled from the gradient update.

class AdamW(Optimizer):
    def __init__ (self, params, lr, weight_decay, betas, eps):
        default = { 
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        }
        super().__init__(params,default)
    
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
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)


                denom = v_hat.sqrt().add(eps)
                update = m_hat / denom + wd * p.data
                
                p.data.add_(update, alpha=-lr)
        
        return loss