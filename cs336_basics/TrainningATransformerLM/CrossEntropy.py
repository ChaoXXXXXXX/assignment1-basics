import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

def softmax(in_features:torch.Tensor,dim:int)->torch.Tensor:
    x = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 直接计算 log_softmax，避免 log(0) 溢出
    # log(softmax(x)) = (x - max) - log(sum(exp(x - max)))
    x = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))
    log_probility = x - log_sum_exp
    batch_size = inputs.shape[0]
    loss = -log_probility[torch.arange(batch_size), targets].mean()
    return loss

