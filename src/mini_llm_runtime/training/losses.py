import torch


def cross_entropy(input, target):
    max_vals = torch.max(input, dim=-1, keepdim=True).values
    shifted = input - max_vals
    logsumexp = (
        torch.log(torch.sum(torch.exp(shifted), dim=-1))
        + max_vals.squeeze(-1)
    )
    batch_indices = torch.arange(input.shape[0], device=input.device)
    target_logits = input[batch_indices, target]
    losses = logsumexp - target_logits
    return losses.mean()
