import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import math


class RMSNorm(nn.Module):
    def __init__(self,d_model: int, W: torch.Tensor, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.W = nn.Parameter(W)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.mean(x**2,dim=-1,keepdim = True) + self.eps)**0.5
        result = (x / rms) * self.W
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 直接接受传入的权重，而不是在内部初始化
        self.W1 = nn.Parameter(W1)  # 这里保持为 Parameter
        self.W3 = nn.Parameter(W3)  # 这里保持为 Parameter
        self.W2 = nn.Parameter(W2)  # 这里保持为 Parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 SiLU(x @ W1.T) 和 (x @ W3.T)
        x1 = x @ self.W1.T  # (batch_size, d_ff)
        x3 = x @ self.W3.T  # (batch_size, d_ff)
        
        # 计算 SiLU 激活值，并进行按元素乘法（gating）
        gated = (x1 * torch.sigmoid(x1)) * x3  # 按元素乘法
        
        # 使用 W2 投影输出
        return gated @ self.W2.T  # (batch_size, d_model)
    


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # inv_freq shape: (d_k/2,)
        k = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2.0 * k / d_k)  # (d_k/2,)

        # positions shape: (max_seq_len,)
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (L,)

        # angles shape: (max_seq_len, d_k/2)
        angles = pos[:, None] * inv_freq[None, :]

        cos_cache = torch.cos(angles).to(dtype if dtype is not None else torch.float32)
        sin_cache = torch.sin(angles).to(dtype if dtype is not None else torch.float32)

        # 这些是固定常量，不训练；persistent=False 表示不进 state_dict（按作业建议）
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)

        cos = self.cos_cache[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]  # (..., seq_len, d_k/2)

        # 拆成偶数维和奇数维（每对做旋转）
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2)

        # 2D 旋转： (a', b') = (a cos - b sin, a sin + b cos)
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # 交错合并回 (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out
def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)



def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor| None = None) -> torch.Tensor:
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same last dimension (d_k)")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same number of keys")
    d_k = q.shape[-1]
    scores = q.to(torch.float32) @ k.to(torch.float32).transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(~mask, -float("inf"))
    attention = softmax(scores,dim = -1)

    return attention @ v.to(torch.float32)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def attention(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor|None = None)->torch.Tensor:
        d_k = self.head_dim
        score = q @ k.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            score.masked_fill_(~mask,-float("inf"))
        attention = softmax(score,dim = -1)
        return attention @ v.to(torch.float32)

    def forward(self,x:torch.Tensor, wq:torch.Tensor, wk:torch.Tensor, wv:torch.Tensor,wo:torch.Tensor)->torch.Tensor:

        batch_size, seq_len, _ = x.shape
        q = x @ wq.T # (batch_size, seq_len, d_model) @ (... d_model d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (... d_model d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (... d_model d_v) -> (batch_size, seq_len, d_v
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        
        q = q.transpose(1,2) #(batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        #创建mask 防止当前的token看答案
        mask = torch.tril(torch.ones((seq_len,seq_len),device = x.device,dtype = torch.bool))

        attention = self.attention(q,k,v,mask) #(batch_size, n_heads, seq_len, head_dim)
        out = attention.transpose(1,2)       #(batch_size, seq_len, n_heads, head_dim)
        out = out.contiguous().view(batch_size,seq_len,self.d_model)
        out = out @ wo.T
        return out

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device, dtype)

    def attention(self,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor|None = None)->torch.Tensor:
        d_k = Q.shape[-1] # head_dim
        score = Q @ K.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            score.masked_fill_(~mask,-float("inf"))
        attention = softmax(score,dim = -1)
        return attention @ V

    def forward(self,x:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,wv:torch.Tensor,wo:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = x @ wq.T # (batch_size, seq_len, d_model) @ (... d_model d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (... d_model d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (... d_model d_v) -> (batch_size, seq_len, d_v)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        
        q = q.transpose(1,2) #(batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        # token_positions: (batch, seq) → (batch, n_heads, seq)
        tp = token_positions.unsqueeze(1).expand(-1, self.n_heads, -1)


        q = self.rope(q,tp)
        k = self.rope(k,tp)

        #mask
        mask = torch.tril(torch.ones((seq_len,seq_len),device = x.device,dtype = torch.bool))
        attention = self.attention(q,k,v,mask)
        out = attention.transpose(1,2)       #(batch_size, seq_len, n_heads, head_dim)
        out = out.contiguous().view(batch_size,seq_len,self.d_model)
        out = out @ wo.T
        return out

        
            
            
                



    

        
