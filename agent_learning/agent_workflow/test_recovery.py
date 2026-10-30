from unittest.mock import patch
import recovery
from agent_state import create_agent_state, start_task
from planner import attach_plan
from executor import execute_plan
import json

def make_state():

    state = create_agent_state("查询订单1001的状态")

    plan = {
        "steps": [
            {
                "id": 1,
                "description": "查询订单1001状态",
                "tool_name": "get_order_status",
                "arguments": {"order_id": 1001},
                "depends_on": []
            }
        ]
    }

    attach_plan(state, plan)
    start_task(state)

    return state


# ========= 实验一：第一次超时，第二次成功 =========

state = make_state()
with patch.object(
    recovery,
    "execute_tool",
    side_effect=[
        TimeoutError("模拟连接超时"),
        {"order_id": 1001, "status": "已完成"}
    ]
) as mock_tool:

    execute_plan(state)
    assert mock_tool.call_count == 2

assert state["status"] == "running"
assert state["plan"][0]["status"] == "completed"
assert state["tool_attempt_count"] == 2
assert state["tool_call_count"] == 1

print("实验一通过：超时后成功恢复")


# ========= 实验二：参数错误，不应重试 =========

state = make_state()

with patch.object(
    recovery,
    "execute_tool",
    side_effect=ValueError("模拟非法参数")
) as mock_tool:

    execute_plan(state)
    assert mock_tool.call_count == 1

assert state["status"] == "failed"
assert state["plan"][0]["status"] == "failed"
assert state["tool_attempt_count"] == 1
assert state["current_step_index"] == 0

print("实验二通过：非法参数立即终止")


# ========= 实验三：连续两次超时 =========

state = make_state()

with patch.object(
    recovery,
    "execute_tool",
    side_effect=TimeoutError("模拟持续超时")
) as mock_tool:

    execute_plan(state)
    assert mock_tool.call_count == 2

assert state["status"] == "failed"
assert state["plan"][0]["status"] == "failed"
assert state["tool_attempt_count"] == 2
assert state["current_step_index"] == 0

print("实验三通过：重试次数耗尽后停止")


# ========= 实验四：整个任务的预算已耗尽 =========

state = make_state()

# 模拟该任务此前已经消耗了全部工具尝试预算
state["tool_attempt_count"] = 5

with patch.object(
    recovery,
    "execute_tool",
    side_effect=AssertionError("工具不应该被调用")
) as mock_tool:

    execute_plan(state)
    assert mock_tool.call_count == 0

assert state["status"] == "failed"
assert state["tool_attempt_count"] == 5
assert state["tool_call_count"] == 0

print("实验四通过：预算耗尽时没有执行工具")


print("\n5.2.7 四组恢复测试通过")



def summarize_execution(state):
    steps = state["plan"]

    return {
        "task_status": state["status"],
        "completed_steps": sum(
            step["status"] == "completed"
            for step in steps
        ),
        "failed_steps": sum(
            step["status"] == "failed"
            for step in steps
        ),
        "skipped_steps": sum(
            step["status"] == "skipped"
            for step in steps
        ),
        "tool_attempt_count": state["tool_attempt_count"],
        "tool_result_count": state["tool_call_count"],
        "error": state["error"]
    }
print("\n\n第四组：")
print(json.dumps(summarize_execution(state),ensure_ascii=False,indent=2))