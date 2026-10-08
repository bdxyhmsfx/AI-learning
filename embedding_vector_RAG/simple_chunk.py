def text_split(text,chunk_size,chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text),chunk_size + start)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end-chunk_overlap
    return chunks

if __name__ == "__main__":
    text = """
    Transformer 是一种基于 Attention 机制的神经网络架构。
    Self-Attention 可以让每个 token 与其他 token 建立联系。
    LoRA 是一种参数高效微调方法。
    RAG 可以让大模型使用外部知识。
    """

    print(text_split(text,30,10))