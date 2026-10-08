import torch
from torch import nn
from d2l import torch as d2l

def transpose_qkv(X, num_heads):
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X, num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention=d2l.DotProductAttention(dropout)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)
    def forward(self,queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values=transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            valid_lens=torch.repeat_interleave(valid_lens,self.num_heads,dim=0)
        output=self.attention(queries,keys,values,valid_lens)
        output_cat=transpose_output(output,self.num_heads)
        return self.W_o(output_cat)

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class TransformBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size, num_hiddens, num_heads,normalized_shape,
                 ffn_num_hiddens, dropout):
        super().__init__()
        self.attention=MultiHeadAttention(key_size,query_size,value_size,num_hiddens, num_heads, dropout)
        self.addnorm1=AddNorm(normalized_shape,dropout)
        self.addnorm2=AddNorm(normalized_shape,dropout)
        self.ffn=PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
    def forward(self, X,valid_lens):
        Y=self.attention(X,X,X,valid_lens)
        X=self.addnorm1(X,Y)
        Y=self.ffn(X)
        X=self.addnorm2(X,Y)
        return X

batch_size = 2
num_steps = 4
num_hiddens = 8
num_heads = 2

block = TransformBlock(
    key_size=8,
    query_size=8,
    value_size=8,
    num_hiddens=8,
    num_heads=2,
    normalized_shape=8,
    ffn_num_hiddens=16,
    dropout=0.0
)

X = torch.randn(batch_size, num_steps, num_hiddens)

# GPT/Decoder风格的因果有效长度
valid_lens = torch.arange(
    1, num_steps + 1
).repeat(batch_size, 1)

print("X.shape =", X.shape)
print("valid_lens =")
print(valid_lens)

Y = block(X, valid_lens)

print("Y.shape =", Y.shape)