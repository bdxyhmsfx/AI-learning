
import json
from agent_state import create_agent_state, start_task
from planner import attach_plan, generate_plan

state = create_agent_state("先查询订单1001的状态，再计算12.5和8的乘积")
manual_plan = {
    "steps": [
        {
            "id": 1,
            "description": "查询订单1001的状态",
            "tool_name": "get_order_status",
            "arguments": {"order_id": 1001},
            "depends_on": []
        },
        {
            "id": 2,
            "description": "计算12.5和8的乘积",
            "tool_name": "multiply",
            "arguments": {"a": 12.5, "b": 8},
            "depends_on": [1]
        }
    ]
}
attach_plan(state, manual_plan)
assert len(state["plan"]) == 2
assert state["plan"][0]["status"] == "pending"
assert state["plan"][1]["depends_on"] == [1]
assert state["current_step_index"] == 0
start_task(state)

print(json.dumps(state["plan"],ensure_ascii=False,indent=2))
print("手动计划测试通过")

state = create_agent_state(
    "先查询订单1001的状态，然后计算12.5和8的乘积"
)

plan_data = generate_plan(state["user_query"])

print("模型原始计划：")
print(json.dumps(plan_data, ensure_ascii=False, indent=2))

attach_plan(state, plan_data)

print("校验后的计划：")
print(json.dumps(state["plan"], ensure_ascii=False, indent=2))