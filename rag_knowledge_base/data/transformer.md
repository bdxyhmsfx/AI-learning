# Transformer

Transformer 是一种基于 Attention 机制的神经网络架构。

与传统 RNN 不同，Transformer 不需要按照时间顺序逐个处理 token，因此在训练时具有较好的并行计算能力。

## Token Embedding

输入文本首先经过 Tokenizer，被转换成一系列 Token ID。

例如：

```text
"我喜欢学习AI"
↓
Tokenizer
↓
[Token1, Token2, Token3, ...]