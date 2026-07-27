import math


def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if it < warmup_iters:
        lr = max_learning_rate * (it / warmup_iters)

    elif warmup_iters <= it <= cosine_cycle_iters:
        process = (it - warmup_iters) / (
            cosine_cycle_iters - warmup_iters
        )
        lr = min_learning_rate + 0.5 * (
            max_learning_rate - min_learning_rate
        ) * (1 + math.cos(math.pi * process))

    else:
        lr = min_learning_rate
    return lr
