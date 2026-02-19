from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def scheduler(t:int,alpha_max:float,alpha_min:float,T_w:int,T_c:int)->float:
    if t<=T_w:
        alpha_t = alpha_max * (t / T_w)
    elif t >= T_w and t <= T_c:
        alpha_t = alpha_min + 1 / 2 * (alpha_max - alpha_min) * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))
    else:
        alpha_t = alpha_min

    alpha_min = alpha_t
    return alpha_t
        

