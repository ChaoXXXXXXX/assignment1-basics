from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Given a model and a maximum norm, clip the gradients of the model's parameters to the maximum norm.
    """
    sum = 0
    for param in parameters:
        if param.grad is not None:
            sum += torch.sum(torch.square(param.grad))
    sum = torch.sqrt(sum)
    if sum > max_l2_norm:
        scale = max_l2_norm / (sum + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad *= scale
            
    return
