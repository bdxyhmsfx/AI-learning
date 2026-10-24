import math


def multiply(a, b):
    return {
        "a": a,
        "b": b,
        "product": a * b
    }


ORDERS = {
    1001: "已完成",
    1002: "待支付",
    1003: "已取消"
}


def get_order_status(order_id):
    return {
        "order_id": order_id,
        "status": ORDERS.get(order_id, "订单不存在")
    }


TOOL_REGISTRY = {
    "multiply": multiply,
    "get_order_status": get_order_status
}


def execute_tool(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"不允许调用的工具：{tool_name}")
    if not isinstance(arguments, dict):
        raise TypeError("arguments 必须是字典")
    if tool_name == "multiply":
        if set(arguments) != {"a", "b"}:
            raise ValueError("multiply 必须提供 a、b 两个参数")
        for key in ("a", "b"):
            value = arguments[key]
            if type(value) not in (int, float):
                raise TypeError(f"{key}必须是数值")
            if not math.isfinite(value):
                raise ValueError(f"{key}必须是有限数值")
            if abs(value) > 1_000_000:
                raise ValueError(f"{key}超出允许范围")
    elif tool_name == "get_order_status":
        if set(arguments) != {"order_id"}:
            raise ValueError("必须提供 order_id")
        if type(arguments["order_id"]) is not int:
            raise TypeError("order_id 必须是整数")
        if arguments["order_id"] <= 0:
            raise ValueError("订单ID必须大于0")
    function = TOOL_REGISTRY[tool_name]
    result = function(**arguments)
    return result
