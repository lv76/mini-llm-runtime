import torch


def silu(in_features):
    return in_features * torch.sigmoid(in_features)


def linear(
    d_in,
    d_out,
    weights,
    in_features,
):
    out = torch.matmul(in_features, weights.t())

    return out


def embedding(
    vocab_size,
    d_model,
    embedding_weight,
    token_ids,
):
    assert embedding_weight.shape == (vocab_size, d_model)

    return embedding_weight[token_ids]


def swiglu(
    d_model,
    d_ff,
    w1_weight,
    w2_weight,
    w3_weight,
    in_features,
):
    x_w1 = torch.matmul(in_features, w1_weight.t())
    x_w3 = torch.matmul(in_features, w3_weight.t())

    silu_x_w1 = silu(x_w1)
    hidden = silu_x_w1 * x_w3

    output = torch.matmul(hidden, w2_weight.t())

    return output


def rmsnorm(d_model, eps, weights, in_features):
    rms = torch.sqrt((in_features**2).mean(dim=-1, keepdim=True)) + eps

    return weights * (in_features / rms)
