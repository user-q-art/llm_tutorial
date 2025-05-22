### GQA（Grouped Query Attention）
标准的 Multi-Head Attention (MHA) 中：

每个 head 都有独立的 Query（Q）、Key（K）、Value（V）。
- 这带来了强大的建模能力，但也导致了：
- 更高的参数量
- 更大的内存占用（尤其是在推理时缓存 K/V）
- 更高的计算成本

为了解决这些问题，研究人员提出了多种简化方案：
| attention module | architecture | features |
|------------------| ------------ | -------- |
| MHA（多头注意力）| Q/K/V 各自独立 head | 表达能力强，但资源消耗大 |
| MQA（Multi-Query Attention）| 多个 Q heads，1 个 K/V head	| 极致效率优化 |
| GQA（Grouped Query Attention）| 多个 Q 分组共享同一组 K/V head | 平衡表达力与效率 |

GQA 工作流程图解
```bash
Q heads: [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]  
分成 4 组：(Q0,Q1), (Q2,Q3), (Q4,Q5), (Q6,Q7)

每组共享一个 K 和 V head：

K heads: [K0, K1, K2, K3]
V heads: [V0, V1, V2, V3]
```

具体实现
```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = group_size
        self.num_groups = num_heads // group_size

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % group_size == 0, "num_heads must be divisible by group_size"

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, attn_mask=None):
        B, L, _ = q.shape

        # Project Q, K, V
        q = self.q_proj(q).view(B, L, self.num_heads, self.head_dim)     # [B, L, H, D]
        k = self.k_proj(k).view(B, L, self.num_groups, self.head_dim)   # [B, L, G, D]
        v = self.v_proj(v).view(B, L, self.num_groups, self.head_dim)   # [B, L, G, D]

        # Repeat K/V to match the number of query heads per group
        k = k.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)       # [B, L, g, G, D]
        v = v.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)
        k = k.flatten(2, 3)                                              # [B, L, H, D]
        v = v.flatten(2, 3)

        # Scaled Dot Product Attention
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)  # [B, H, L, D]
        v = v.transpose(1, 2)  # [B, H, L, D]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(-1)

        x = attn @ v  # [B, H, L, D]
        x = x.transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, E]
        x = self.out_proj(x)
        return x
```

### SwiGLU

收敛更快，引入门控机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### RoPE

示例代码
```python
import torch
import math

def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute rotation frequencies for RoPE.
    
    Args:
        dim (int): Head dimension (must be even).
        end (int): Maximum sequence length to precompute.
        theta (float): Base frequency scaling factor.

    Returns:
        freqs_complex: (end, dim // 2)
    """
    assert dim % 2 == 0, "dim must be even"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # outer product
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_complex

def apply_rotary_emb(x: torch.Tensor, freqs_complex: torch.Tensor):
    """
    Apply rotary embeddings to a tensor.

    Args:
        x: (B, H, L, D)
        freqs_complex: (L, D)

    Returns:
        x_out: same shape as x
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)

# 示例使用
if __name__ == "__main__":
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64

    # 预计算旋转频率
    freqs_complex = precompute_freqs(head_dim, seq_len)

    # 创建随机 Query 矩阵
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 应用 RoPE
    q_with_rope = apply_rotary_emb(q, freqs_complex)

    print("Output shape:", q_with_rope.shape)  # [2, 8, 10, 64]
```

### RMSNorm with pre-normalization

```python
import torch
import torch.nn as nn

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm module.

        Args:
            dim (int): The dimension to normalize over (usually the feature/channel dimension).
            eps (float): Small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)

        Returns:
            torch.Tensor: Output tensor with RMS normalization applied.
        """
        # 计算 RMS（Root Mean Square）
        rms = x.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        # 应用归一化和可学习缩放
        return x * rms * self.weight


class RMSNormBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # Pre-Norm + Multi-Head Attention
        x = x + self.attn(self.norm(x), self.norm(x), self.norm(x))[0]

        # Pre-Norm + Feed Forward
        x = x + self.ffn(self.norm(x))

        return x
```