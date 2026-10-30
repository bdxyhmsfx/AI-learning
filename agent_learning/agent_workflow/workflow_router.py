from typing import Literal
Route = Literal[
    "order_only",
    "multiply_only",
    "sequential",
    "conditional",
    "unsupported"
]

def route_query(query:str)->Route:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    has_order = "订单" in query
    has_multiply = any(word in query for word in ("计算", "乘积", "乘法"))
    has_condition = any(word in query for word in("只有", "仅当", "如果"))
    if has_order and has_multiply:
        if has_condition:
            return "conditional"
        return "sequential"
    if has_order:
        return "order_only"
    if has_multiply:
        return "multiply_only"
    return "unsupported"