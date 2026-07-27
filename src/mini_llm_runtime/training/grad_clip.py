import torch


def gradient_clipping(parameters, max_l2_norm):
    grads = []

    for p in parameters:
        if p.grad is not None and p.requires_grad:
            grads.append(p.grad)

    total_sum = 0.0
    for g in grads:
        total_sum += g.norm(2) ** 2

    total_norm = torch.sqrt(total_sum)

    clip_num = max_l2_norm / (total_norm + 1e-6)

    if clip_num < 1:
        for g in grads:
            g.mul_(clip_num)
