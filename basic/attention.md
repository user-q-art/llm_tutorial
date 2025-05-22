### Different to traditional transformer，GPT-style LLM use decoder-only structure with masked multi-head self-attention in the first layer to model causal relationship in token prediction.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 线性变换用于生成 Q, K, V
        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)

        # 最后的输出线性层
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        """
        参数：
            values: [N, value_len, embed_size]
            keys:   [N, key_len, embed_size]
            query:  [N, query_len, embed_size]
            mask:   [N, 1, 1, query_len] (Optional)，用于屏蔽 future tokens
        """
        N = query.shape[0]
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = query.shape[1]

        # 拆分多头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 转换为 [N, heads, len, head_dim]
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Scaled Dot-Product: (N, heads, query_len, key_len)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        # 如果有 mask，应用 mask（例如未来位置置为 -inf）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # softmax over key_len
        attention = torch.softmax(energy, dim=-1)  # [N, heads, query_len, key_len]

        # 加权求和
        out = torch.matmul(attention, values)  # [N, heads, query_len, head_dim]

        # 合并多头
        out = out.transpose(1, 2).contiguous()  # [N, query_len, heads, head_dim]
        out = out.reshape(N, query_len, self.embed_size)  # [N, query_len, embed_size]

        # 输出线性变换
        out = self.fc_out(out)

        return out, attention
```

usage sample:

```python
# 假设 batch_size=32, seq_length=10, embedding_dim=512, num_heads=8
x = torch.randn(32, 10, 512)

# 创建一个上三角矩阵作为 mask（屏蔽掉未来 token）
mask = torch.tril(torch.ones(1, 1, 10, 10))  # 下三角矩阵，形状 [1, 1, 10, 10]

# 初始化注意力模块
attention = MaskedSelfAttention(embed_size=512, heads=8)

# 前向传播
out, attn_weights = attention(x, x, x, mask)

print("Output shape:", out.shape)  # [32, 10, 512]
```

hints:
1、mask 是一个布尔型张量，值为 0 的位置会被替换成 -inf，softmax 后这些位置权重趋近于 0。
2、在 Transformer 解码器中，通常只在 decoder 的第一个 Multi-Head Attention 层使用这种 mask。
3、构建完整的 Transformer Decoder Layer，可以将该模块与 LayerNorm、FeedForward 等组合起来。