from rag_knowledge_base.generation.citation import extract_citation_ids, validate_citations

citations = {
    1: {
        "source": "attention.pdf",
        "page": 3,
        "chunk_id": "abc123"
    },
    2: {
        "source": "transformer.md",
        "page": None,
        "chunk_id": "def456"
    }
}

answers = [
    "Multi-Head Attention 可以学习不同的信息。[1]",

    "Attention 是重要机制。[1] "
    "Transformer 可以处理序列信息。[2] "
    "这些机制都很重要。[1]",

    "Attention 使用多个注意力头。[1] "
    "这是另一个没有有效来源的结论。[9]",

    "这是一个没有提供引用编号的回答。"
]

for answer in answers:
    result = validate_citations(answer, citations)
    print("\n" + "=" * 60)
    print("Answer：", answer)
    print("提取编号：",extract_citation_ids(answer))
    print("有效来源：",result["used_citations"])
    print("无效编号：",result["invalid_ids"])
    print("是否包含引用：",result["has_citations"])
