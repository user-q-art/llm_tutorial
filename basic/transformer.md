# Simple Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        # 定义 Q, K, V 的线性层（这里简单起见使用相同的 embed_dim）
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, embed_dim)
        return: shape (batch_size, seq_len, embed_dim)
        """
        Q = self.query(x)   # (B, L, D)
        K = self.key(x)     # (B, L, D)
        V = self.value(x)   # (B, L, D)

        # 计算注意力得分：(B, L, D) @ (B, D, L) -> (B, L, L)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = attn_weights / (self.embed_dim ** 0.5)  # 缩放

        # Softmax 归一化
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 加权求和 Value
        output = torch.matmul(attn_weights, V)  # (B, L, L) @ (B, L, D) -> (B, L, D)

        return output

```

### (Enhanced) Multi-Head Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim 必须是 num_heads 的整数倍"

        # 定义 QKV 的线性变换
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 输出线性层
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, embed_dim)
        return: shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()

        # 线性变换 + reshape 成多头形式
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # (B, L, H, D_head)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转换为 (B, H, L, D_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力得分：(B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        # Softmax 归一化
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 加权求和：(B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
        output = torch.matmul(attn_weights, V)

        # 恢复形状：(B, H, L, D) -> (B, L, H, D) -> (B, L, E)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 输出线性变换
        output = self.out(output)

        return output
```

### （Enhanced）Multi-Head Self-Attention + LayerNorm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(TransformerBlock, self).__init__()

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 第一步：多头注意力 + 残差连接
        attn_output = self.attn(x)
        x = x + attn_output
        x = self.norm1(x)

        # 后续可添加 FFN（下一部分添加）

        return x
```
一个标准的LayerNorm实现为
```python
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(MyLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 标准化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移
        return self.gamma * x_normalized + self.beta
```

### (Enhanced) Multi-Head Self-Attention + LayerNorm + FFN -> 完整Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),  # 可换成 nn.ReLU()
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 第一步：多头注意力 + 残差连接 + LayerNorm
        x = x + self.attn(x)
        x = self.norm1(x)

        # 第二步：FFN + 残差连接 + LayerNorm
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x
```

### (Enhanced) Pos Enc "词或 patch 在序列中的位置"
```python
PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
# pos 是位置索引（从 0 到 max_len）
# i 是维度索引（从 0 到 d_model）
```