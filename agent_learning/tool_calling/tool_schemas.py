get_order_status_tool = {
    "type": "function",
    "name": "get_order_status",
    "description": "根据订单 ID 查询订单当前状态，仅用于读取信息。",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "integer",
                "description": "需要查询的订单编号"
            }
        },
        "required": ["order_id"],
        "additionalProperties": False
    },
    "strict": True
}

multiply_tool = {
    "type":"function",
    "name":"multiply",
    "description":"计算输入的两个数的乘积",
    "parameters":{
        "type":"object",
        "properties":{
            "a":{"type":"number"},
            "b":{"type":"number"},
        },
        "required":["a","b"],
        "additionalProperties":False
    },
    "strict":True
}