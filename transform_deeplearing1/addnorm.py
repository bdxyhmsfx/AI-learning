import torch
from torch import nn
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
a=torch.ones((2, 3, 4))
b=torch.ones((2, 3, 4))
print(a)
print(b)
print(add_norm(a, b))