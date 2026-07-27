import torch
import math
import numpy as np

def get_batch(dataset, batch_size, context_length, device):

    start = np.random.randint(
        0, 
        len(dataset) - context_length, 
        size = batch_size
    )

    x = np.stack([dataset[i : i + context_length] for i in start])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in start])

    x = torch.tensor(x, dtype=torch.long, device = device)
    y = torch.tensor(y, dtype=torch.long, device = device)

    return x, y
