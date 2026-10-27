import json
from agent_state import create_agent_state, start_task, finish_task
from planner import attach_plan
from executor import execute_plan
from planner import generate_plan
state = create_agent_state("先查询订单1001的状态，再计算12.5和8的乘积")
# 手工测试
# manual_plan = {
#     "steps": [
#         {
#             "id": 1,
#             "description": "查询订单1001的状态",
#             "tool_name": "get_order_status",
#             "arguments": {"order_id": 1001},
#             "depends_on": []
#         },
#         {
#             "id": 2,
#             "description": "计算12.5和8的乘积",
#             "tool_name": "multiply",
#             "arguments": {"a": 12.5, "b": 8},
#             "depends_on": [1]
#         }
#     ]
# }
manual_plan = generate_plan(state["user_query"])
attach_plan(state, manual_plan)
start_task(state)
execute_plan(state)
assert state["plan"][0]["status"] == "completed"
assert state["plan"][1]["status"] == "completed"
assert state["current_step_index"] == 2
assert state["tool_call_count"] == 2
assert state["current_step"] == "awaiting_final_answer"
assert state["status"] == "running"
order = state["plan"][0]["result"]
multiply = state["plan"][1]["result"]
answer = (
    f"订单{order['order_id']}状态为{order['status']}，"
    f"乘积为{multiply['product']}。"
)
finish_task(state, answer)
assert state["status"] == "completed"
print("\n===== 最终 State =====")
print(json.dumps(state, ensure_ascii=False, indent=2))

# 测试非法参数
# bad_state = create_agent_state("测试非法乘法参数")
# bad_plan = {
#     "steps": [
#         {
#             "id": 1,
#             "description": "测试非法参数",
#             "tool_name": "multiply",
#             "arguments": {"a": "hello", "b": 8},
#             "depends_on": []
#         }
#     ]
# }
# attach_plan(bad_state, bad_plan)
# start_task(bad_state)
# execute_plan(bad_state)
#
# print(bad_state["status"])
# print(bad_state["plan"][0]["status"])
# print(bad_state["error"])

# 测试前一个步骤未完成
# state = create_agent_state("先查询订单1001的状态，再计算12.5和8的乘积")
# manual_plan = {
#     "steps": [
#         {
#             "id": 1,
#             "description": "查询订单1001的状态",
#             "tool_name": "get_order_status",
#             "arguments": {"order_id": 1001},
#             "depends_on": []
#         },
#         {
#             "id": 2,
#             "description": "计算12.5和8的乘积",
#             "tool_name": "multiply",
#             "arguments": {"a": 12.5, "b": 8},
#             "depends_on": [1]
#         }
#     ]
# }
# attach_plan(state, manual_plan)
# start_task(state)
# execute_plan(state)
# print(state["status"])
# print(state["plan"][0]["status"])
# print(state["error"])
