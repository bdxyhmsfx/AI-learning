def build_context(reranked_results, max_chars=4000):
    if max_chars < 0:
        raise ValueError("max_chars 必须大于 0")
    context_parts = []
    citations = {}
    seen_ids = set()
    current_length = 0
    for result in reranked_results:
        text = (result.get("text") or "").strip()
        if not text:
            continue
        chunk_id = result.get("chunk_id")
        source = result.get("source") or "未知来源"
        page = result.get("page")
        unique_key = (chunk_id if chunk_id else (source,page,text))
        if unique_key in seen_ids:
            continue
        citation_id = len(citations)+1
        header=f"[{citation_id}] 来源:{source}"
        if page is not None:
            header+=f" | 第{page}页"
        block=f"{header}\n内容：{text}"
        additional_length=len(block)
        if context_parts:
            additional_length+=2
        if additional_length+current_length>max_chars:
            continue
        context_parts.append(block)
        current_length+=additional_length
        citations[citation_id]={
            "source":source,
            "page":page,
            "chunk_id":chunk_id,
        }
        seen_ids.add(unique_key)
    context="\n\n".join(context_parts)
    return context,citations