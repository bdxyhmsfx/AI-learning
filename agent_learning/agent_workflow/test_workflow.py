
from agent_learning.agent_workflow.workflow_router import route_query
from agent_learning.agent_workflow.workflow_controller import run_workflow


test_cases = [
    (
        "查询订单1001的状态",
        1001,
        "order_only",
        ["completed"],
        1
    ),
    (
        "计算12.5和8的乘积",
        None,
        "multiply_only",
        ["completed"],
        1
    ),
    (
        "先查询订单1002的状态，再计算12.5和8的乘积",
        1002,
        "sequential",
        ["completed", "completed"],
        2
    ),
    (
        "只有订单1001已完成才计算12.5和8的乘积",
        1001,
        "conditional",
        ["completed", "completed"],
        2
    ),
    (
        "只有订单1002已完成才计算12.5和8的乘积",
        1002,
        "conditional",
        ["completed", "skipped"],
        1
    ),
    (
        "解释一下数据库索引是什么",
        None,
        "unsupported",
        [],
        0
    )
]


for query, order_id, expected_route, expected_statuses, expected_calls in test_cases:

    print("\n" + "=" * 50)
    print("用户问题：", query)

    route = route_query(query)
    assert route == expected_route

    state = run_workflow(
        query,
        order_id=order_id
    )

    actual_statuses = [
        step["status"] for step in state["plan"]
    ]

    assert actual_statuses == expected_statuses
    assert state["tool_call_count"] == expected_calls

    if expected_route == "unsupported":
        assert state["status"] == "failed"
    else:
        assert state["status"] == "completed"

    print("最终状态：", state["status"])
    print("步骤状态：", actual_statuses)
    print("工具调用次数：", state["tool_call_count"])
    print("最终回答：", state["final_answer"])


print("\n5.2.6 全部测试通过")
