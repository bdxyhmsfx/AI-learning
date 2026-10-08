import jieba
import numpy as np
from rank_bm25 import BM25Okapi

def tokenize_for_bm25(text):
    text = text.lower()
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens if token.strip()]
    return tokens

def build_bm25_index(chunks):
    tokenized_corpus = []

    for chunk in chunks:
        text = chunk["text"]
        tokens = tokenize_for_bm25(text)
        tokenized_corpus.append(tokens)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def bm25_search(query,chunks,bm25,k=3):
    query_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for index in top_indices:
        chunk = chunks[index]
        result = chunk.copy()
        result["score"] = float(scores[index])
        result["retriever"] = "bm25"
        results.append(result)
    return results