from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            beta1, beta2 = betas

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化一阶和二阶矩
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['step'] = 0

                m, v = state['m'], state['v']

                # 时间步 +1
                state['step'] += 1
                t = state['step']

                # 更新一阶矩 (mean of gradients)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 更新二阶矩 (mean of squared gradients)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Adam 更新
                p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                # AdamW 权重衰减（解耦，直接衰减参数）
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

        return loss
