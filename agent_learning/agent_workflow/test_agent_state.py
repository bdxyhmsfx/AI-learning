import json
from agent_state import create_agent_state,start_task,record_round,record_tool_result,finish_task,fail_task

state = create_agent_state("先查询订单1001的状态，再计算12.5和8的乘积")
print("初始状态：", state["status"])
start_task(state)
record_round(state)
record_tool_result(state,"get_order_status",{"order_id": 1001, "status": "已完成"})
record_round(state)
record_tool_result(state,"multiply",{"a": 12.5, "b": 8, "product": 100.0})
record_round(state)
finish_task(state,"订单1001已完成，乘积为100。")

assert state["status"] == "completed"
assert state["round_count"] == 3
assert state["tool_call_count"] == 2
assert len(state["tool_results"]) == 2

print("\n===== 正常任务最终状态 =====")
print(json.dumps(state, ensure_ascii=False, indent=2))


failed_state = create_agent_state("查询订单9999")
start_task(failed_state)
record_round(failed_state)
fail_task(failed_state, "数据库连接失败")
assert failed_state["status"] == "failed"
assert failed_state["error"] == "数据库连接失败"
print("\n===== 失败任务 =====")
print(failed_state["status"])
print(failed_state["error"])