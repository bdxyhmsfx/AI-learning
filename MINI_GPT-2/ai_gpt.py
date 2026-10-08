import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# 1. 基本参数
# ============================================================

torch.manual_seed(1337)

batch_size = 32       # 一次训练多少条序列
block_size = 64       # 每条序列最多多少个 token
max_iters = 3000         # 训练多少次
eval_interval = 300   # 每隔多少次打印一次 loss
learning_rate = 3e-4
eval_iters = 100

n_embd = 128          # 每个 token 的隐藏维度
n_head = 4            # Multi-Head Attention 的 head 数量
n_layer = 4           # Transformer Block 数量
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("device:", device)


# ============================================================
# 2. 读取文本
# ============================================================

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# ============================================================
# 3. 构造最简单的字符级 Tokenizer
# ============================================================

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("vocab_size:", vocab_size)

# 字符 -> token ID
stoi = {ch: i for i, ch in enumerate(chars)}

# token ID -> 字符
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join([itos[i] for i in ids])


# 把整个文本转换成 token ID
data = torch.tensor(
    encode(text),
    dtype=torch.long
)

# 90% 训练，10% 验证
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# ============================================================
# 4. 生成一个 batch
# ============================================================

def get_batch(split):

    data = train_data if split == 'train' else val_data

    # 随机选择 batch_size 个起点
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # 输入
    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    # 正确答案，比 x 整体后移一个 token
    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# ============================================================
# 5. 单个 Self-Attention Head
# ============================================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # causal mask
        self.register_buffer(
            'tril',
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        # [B,T,C] -> [B,T,head_size]
        k = self.key(x)
        q = self.query(x)

        # QK^T
        # [B,T,H] @ [B,H,T]
        # -> [B,T,T]
        wei = q @ k.transpose(-2, -1)

        # scaled dot-product attention
        wei = wei * (k.shape[-1] ** -0.5)

        # causal mask
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # attention score -> attention weight
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        # V
        v = self.value(x)

        # Attention Weight × V
        # [B,T,T] @ [B,T,H]
        # -> [B,T,H]
        out = wei @ v

        return out


# ============================================================
# 6. Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        # 拼接所有 head 后再做一次线性变换
        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # 每个 head：
        # [B,T,head_size]
        #
        # 在最后一维拼起来：
        # [B,T,n_embd]

        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        out = self.proj(out)
        out = self.dropout(out)

        return out


# ============================================================
# 7. FFN
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(

            # 先扩大
            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.ReLU(),

            # 再缩回来
            nn.Linear(
                4 * n_embd,
                n_embd
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 8. Transformer Block
# ============================================================

class GPTBlock(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Pre-Norm + Residual
        x = x + self.sa(self.ln1(x))

        # Pre-Norm + Residual
        x = x + self.ffwd(self.ln2(x))

        return x


# ============================================================
# 9. 完整 GPT 模型
# ============================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Token ID -> token embedding
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        # position -> position embedding
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        # 堆叠多个 Transformer Block
        self.blocks = nn.Sequential(
            *[
                GPTBlock(n_embd, n_head)
                for _ in range(n_layer)
            ]
        )

        # 最后一次 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # hidden representation -> vocab logits
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        tok_emb = self.token_embedding_table(idx)

        # [B,T]
        # ->
        # [B,T,n_embd]

        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        pos = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding_table(pos)

        # [T,n_embd]

        # ----------------------------------------------------
        # Token + Position
        # ----------------------------------------------------

        x = tok_emb + pos_emb

        # [B,T,n_embd]

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        x = self.blocks(x)

        # [B,T,n_embd]

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.ln_f(x)

        # ----------------------------------------------------
        # LM Head
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # [B,T,vocab_size]

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        if targets is None:

            loss = None

        else:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss


    # ========================================================
    # 10. 文本生成
    # ========================================================

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # 只保留模型允许的最大上下文
            idx_cond = idx[:, -block_size:]

            # 模型预测
            logits, _ = self(idx_cond)

            # 只需要最后一个位置
            logits = logits[:, -1, :]

            # logits -> probability
            probs = F.softmax(
                logits,
                dim=-1
            )

            # 按概率采样一个 token
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # 接到原序列后面
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


# ============================================================
# 11. 创建模型
# ============================================================

model = GPTLanguageModel().to(device)

print(
    "参数量:",
    sum(p.numel() for p in model.parameters()) / 1e6,
    "M"
)


# ============================================================
# 12. 评估 loss
# ============================================================

@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ['train', 'val']:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# ============================================================
# 13. 优化器
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 14. 训练
# ============================================================

for step in range(max_iters):

    # 定期打印训练集和验证集 loss
    if step % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"step {step}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # 获取一个 batch
    xb, yb = get_batch('train')

    # forward
    logits, loss = model(xb, yb)

    # 清梯度
    optimizer.zero_grad(set_to_none=True)

    # backward
    loss.backward()

    # 更新参数
    optimizer.step()


# ============================================================
# 15. 生成文本
# ============================================================

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated_ids = model.generate(
    context,
    max_new_tokens=500
)

print(
    decode(
        generated_ids[0].tolist()
    )
)