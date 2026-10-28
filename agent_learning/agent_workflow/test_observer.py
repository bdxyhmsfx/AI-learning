from agent_learning.agent_workflow.agent_state import create_agent_state, start_task, finish_task
from agent_learning.agent_workflow.planner import attach_plan
from agent_learning.agent_workflow.executor import execute_next_step
from agent_learning.agent_workflow.observer import observe_order_result, replan_after_observation


def run_case(order_id: int):
    query = (
        f"查询订单{order_id}的状态。"
        "只有状态为已完成时，才计算12.5和8的乘积。"
    )
    state = create_agent_state(query)
    plan_data = {
        "steps": [
            {
                "id": 1,
                "description": f"查询订单{order_id}状态",
                "tool_name": "get_order_status",
                "arguments": {"order_id": order_id},
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
    attach_plan(state, plan_data)
    start_task(state)

    while (state["status"] == "running" and state["current_step_index"] < len(state["plan"])):
        index = state["current_step_index"]
        step = state["plan"][index]

        if step["status"] == "skipped":
            state["current_step_index"] += 1
            continue
        success = execute_next_step(state)
        if not success:
            break
        observation = observe_order_result(state)
        if observation is not None:
            replan_after_observation(state, observation)

    if state["status"] != "running":
        return state
    state["current_step"] = "awaiting_final_answer"
    order_result = state["plan"][0]["result"]
    multiply_step = state["plan"][1]

    if multiply_step["status"] == "completed":
        answer = (
            f"订单{order_id}状态为{order_result['status']}，"
            f"乘积为{multiply_step['result']['product']}。"
        )
    elif multiply_step["status"] == "skipped":
        answer = (
            f"订单{order_id}状态为{order_result['status']}，""不满足计算条件，已跳过乘法步骤。")
    else:
        raise RuntimeError("计划未正确结束")
    finish_task(state, answer)
    print("\n最终回答：", state["final_answer"])
    print("步骤状态：", [step["status"] for step in state["plan"]])

    return state


if __name__ == "__main__":
    completed_order = run_case(1001)

    assert completed_order["status"] == "completed"
    assert completed_order["tool_call_count"] == 2
    assert completed_order["plan"][1]["status"] == "completed"
    assert completed_order["observations"][0]["condition_met"]

    pending_order = run_case(1002)

    assert pending_order["status"] == "completed"
    assert pending_order["tool_call_count"] == 1
    assert pending_order["plan"][1]["status"] == "skipped"
    assert not pending_order["observations"][0]["condition_met"]

    print("\n5.2.5 两组测试均通过")
