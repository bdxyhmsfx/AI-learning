from agent_runner import run_agent_workflow


reverse_plan = {
    "steps": [
        {
            "id": 1,
            "description": "计算12.5和8的乘积",
            "tool_name": "multiply",
            "arguments": {"a": 12.5, "b": 8},
            "depends_on": []
        },
        {
            "id": 2,
            "description": "查询订单1001",
            "tool_name": "get_order_status",
            "arguments": {"order_id": 1001},
            "depends_on": [1]
        }
    ]
}

state = run_agent_workflow(
    "先计算12.5和8的乘积，再查询订单1001",
    reverse_plan
)

assert state["status"] == "completed"

assert [
    step["tool_name"] for step in state["plan"]
] == ["multiply", "get_order_status"]

assert state["tool_call_count"] == 2
assert state["tool_attempt_count"] == 2

print(state["final_answer"])
print("实验一通过")

conditional_plan = {
    "steps": [
        {
            "id": 1,
            "description": "查询订单1002",
            "tool_name": "get_order_status",
            "arguments": {"order_id": 1002},
            "depends_on": []
        },
        {
            "id": 2,
            "description": "订单已完成才计算乘积",
            "tool_name": "multiply",
            "arguments": {"a": 12.5, "b": 8},
            "depends_on": [1],
            "run_if": {
                "step_id": 1,
                "status_equals": "已完成"
            }
        }
    ]
}

state = run_agent_workflow(
    "只有订单1002已完成才计算12.5和8的乘积",
    conditional_plan
)

assert state["status"] == "completed"
assert state["plan"][0]["status"] == "completed"
assert state["plan"][1]["status"] == "skipped"

assert state["tool_call_count"] == 1
assert state["observations"][0]["condition_met"] is False

print(state["final_answer"])
print("实验二通过")



invalid_plan = {
    "steps": [
        {
            "id": 1,
            "description": "调用不存在的工具",
            "tool_name": "delete_everything",
            "arguments": {},
            "depends_on": []
        }
    ]
}

state = run_agent_workflow(
    "执行非法工具",
    invalid_plan
)

assert state["status"] == "failed"
assert state["tool_attempt_count"] == 0
assert len(state["plan"]) == 0

print("实验四通过")


from unittest.mock import patch
import recovery

with patch.object(
    recovery,
    "execute_tool",
    side_effect=[
        TimeoutError("模拟超时"),
        {"a": 12.5, "b": 8, "product": 100.0},
        {"order_id": 1001, "status": "已完成"}
    ]
) as mock_tool:

    state = run_agent_workflow(
        "先计算乘积再查询订单",
        reverse_plan
    )

    assert mock_tool.call_count == 3

assert state["status"] == "completed"
assert state["tool_attempt_count"] == 3
assert state["tool_call_count"] == 2

print("实验五通过")


from planner import generate_plan, validate_plan
from agent_runner import run_agent_workflow

query = "只有订单1002已完成，才计算12.5和8的乘积"

plan_data = generate_plan(query)

# 先验证结构，再验证顺序
validate_plan(plan_data)

assert [
    step["tool_name"] for step in plan_data["steps"]
] == ["multiply", "get_order_status"]

state = run_agent_workflow(query, plan_data)

assert state["status"] == "completed"

print(state["final_answer"])