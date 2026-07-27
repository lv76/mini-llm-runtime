import numpy
import torch


def softmax(x, dim = -1):
    max_vals, _ = torch.max(x, dim = dim, keepdim = True)
    x_shift = x - max_vals
    exp_x = torch.exp(x_shift)

    sum_exp = torch.sum(exp_x, dim = dim, keepdim = True)
    softmax_out = exp_x / sum_exp

    return softmax_out

"""
def cross_entropy(input, target):
    probs = softmax(input , dim = -1)
    log_probs = torch.log(probs + 1e-9)
    log_probs_flat = log_probs.view(-1, input.shape[-1]) 

    targets = target.view(-1)
    correct_log_probs = log_probs_flat[torch.arange(len(targets)), targets]

    return -correct_log_probs.mean()
"""

def cross_entropy(input, target):
    max_vals = torch.max(input, dim=-1, keepdim=True).values
    shifted = input - max_vals
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + max_vals.squeeze(-1)
    batch_indices = torch.arange(input.shape[0], device=input.device)
    target_logits = input[batch_indices, target]
    losses = logsumexp - target_logits
    return losses.mean()

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





