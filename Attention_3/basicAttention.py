import math
import torch
import torch.nn.functional as F

def attention(q,k,v):
    scores=q @ k.transpose(1,2)
    scores/=math.sqrt(q.shape[-1])
    weights=F.softmax(scores,dim=-1)
    output=weights @ v
    return output, weights

Q=torch.randn(2,4,8)
K=torch.randn(2,4,8)
V=torch.randn(2,4,8)

outputs, weight = attention(Q,K,V)
print(outputs.shape)
print(weight.shape)