from tool_dispatcher import execute_tool
test_cases = [
    (
        "未知工具",
        "delete_order",
        {"order_id": 1001}
    ),
    (
        "缺少参数",
        "multiply",
        {"a": 12.5}
    ),
    (
        "错误参数类型",
        "multiply",
        {"a": "hello", "b": 8}
    ),
    (
        "额外参数",
        "get_order_status",
        {"order_id": 1001, "delete": True}
    ),
    (
        "非法订单 ID",
        "get_order_status",
        {"order_id": -1}
    ),
    (
        "无穷大参数",
        "multiply",
        {"a": float("inf"), "b": 8}
    )
]
for title,tool_name,arguments in test_cases:
    print(f"\n===== {title} =====")
    try:
        result = execute_tool(tool_name, arguments)
    except (ValueError, TypeError) as e:
        print("PASS：正确拒绝")
        print("错误原因：", e)
    else:
        raise AssertionError(f"{title} 未被正确拒绝：{result}")