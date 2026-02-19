import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn

class EmbeddingModule(nn.Module):
    def __init__(self,num_embddings: int, embdding_dim: int, device : torch.device | None = None,dtype : torch.dtype | None = None):
        super().__init__()
        self.num_embddings = num_embddings
        self.embdding_dim = embdding_dim
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(num_embddings,embdding_dim,device=device,dtype=dtype))
        std = 2 / (num_embddings + embdding_dim) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean = 0, std = std,a = -3.0,b = 3.0)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        return self.W[token_ids]        
        