import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn

class LinearModule(nn.Module):
    def __init__(self,in_features:int,out_features:int,device : torch.device | None = None,dtype : torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))

        std = 2 / (self.in_features + self.out_features) ** 0.5
        torch.nn.init.uniform_(self.W, a = -3*std, b = 3*std)
        

    def forward(self,x:Float[Tensor,"... in_features"]):
        return x @ self.W.T
        
