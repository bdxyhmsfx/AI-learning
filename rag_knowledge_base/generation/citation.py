import re

def extract_citation_ids(answer):
    if not isinstance(answer,str):
        raise TypeError("answer 必须是字符串")
    matches = re.findall(r"\[(\d+)\]",answer)
    citation_ids = []
    for number in matches:
        citation_id=int(number)
        if citation_id not in citation_ids:
            citation_ids.append(citation_id)
    return citation_ids

def validate_citations(answer,citations):
    if not isinstance(citations,dict):
        raise TypeError("citations 必须是字典")
    citation_ids = extract_citation_ids(answer)
    used_citations={}
    invalid_ids=[]
    for citation_id in citation_ids:
        if citation_id not in citations:
            invalid_ids.append(citation_id)
            continue
        used_citations[citation_id]=citations[citation_id].copy()
    return {
        "used_citations": used_citations,
        "invalid_ids": invalid_ids,
        "has_citations": bool(citation_ids)
    }