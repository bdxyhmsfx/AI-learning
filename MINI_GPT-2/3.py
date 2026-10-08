import torch
import math
from torch import nn
from torch.nn import functional as F

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

with open("input.txt",'r',encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s):
    return [stoi[ch] for ch in s]
def decode(idx):
    return ''.join([itos[i] for i in idx])

data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]

def get_batch(split):
    data=train_data if split == 'train' else test_data
    idx = torch.randint(len(data)-block_size,(batch_size,))

    X=torch.stack([data[i:i+block_size] for i in idx])
    Y=torch.stack([data[i+1:i+block_size+1] for i in idx])

    return X.to(device),Y.to(device)

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query=nn.Linear(n_embd,head_size)
        self.key=nn.Linear(n_embd,head_size)
        self.value=nn.Linear(n_embd,head_size)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size,block_size)),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        wei=q @ k.transpose(-1,-2)
        wei=wei/math.sqrt(k.shape[-1])
        wei=wei.masked_fill(
            self.tril[:T,:T]==0,
            float('-inf')
        )
        wei=F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out=wei @ v
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            Head(head_size)
            for _ in range(n_head)
        )
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out=torch.cat([head(x) for head in self.heads],dim=-1)
        out=self.proj(out)
        out=self.dropout(out)
        return out
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class GPTBlock(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.sa=MultiHeadAttention(n_head,n_embd//n_head)
        self.ffn=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table= nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(
            *[GPTBlock(n_embd) for _ in range(n_layer)]
        )
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        embed=self.token_embedding_table(idx)
        pos=torch.arange(T,device=idx.device)
        pos=self.position_embedding_table(pos)
        x=embed+pos
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits_flat=logits.view(B*T,C)
            targets_flat=targets.view(B*T)
            loss=F.cross_entropy(logits_flat,targets_flat)
        return logits,loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,_=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)

        return idx

model = GPT().to(device)

print(
    "参数量:",
    sum(p.numel() for p in model.parameters()) / 1e6,
    "M"
)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for step in range(max_iters):
    if step%eval_interval==0:
        losses=estimate_loss()
        print(
            f"step{step}:"
            f"train loss{losses['train']:.4f}"
            f"val loss{losses['val']:.4f}"
        )
    xb,yb=get_batch("train")
    logits,loss=model(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context=torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

model.eval()

generated_ids = model.generate(
    context,
    max_new_tokens=500
)

print(decode(generated_ids[0].tolist()))