from agent_state import create_agent_state, start_task, finish_task, fail_task
from planner import attach_plan
from executor import execute_next_step, execute_plan
from observer import observe_order_result, replan_after_observation
from workflow_router import route_query


def build_plan_for_route(route: str, order_id: int | None, a: float, b: float) -> dict:
    order_step = {
        "id": 1,
        "description": "查询订单状态",
        "tool_name": "get_order_status",
        "arguments": {"order_id": order_id},
        "depends_on": []
    }
    multiply_step = {
        "id": 1,
        "description": "计算乘积",
        "tool_name": "multiply",
        "arguments": {"a": a, "b": b},
        "depends_on": []
    }
    if route == "order_only":
        return {"steps": [order_step]}
    if route == "multiply_only":
        return {"steps": [multiply_step]}
    if route in ("sequential", "conditional"):
        multiply_step["id"] = 2
        multiply_step["depends_on"] = [1]
        return {"steps": [order_step, multiply_step]}
    raise ValueError(f"不支持的 Route：{route}")


def run_workflow(query: str, order_id: int | None = None, a: float = 12.5, b: float = 8):
    state = create_agent_state(query)
    route = route_query(query)
    print("\n选择的 Route：", route)
    if route == "unsupported":
        fail_task(state, "当前 Workflow 无法处理该任务")
        return state
    if route != "multiply_only" and type(order_id) is not int:
        fail_task(state, "订单任务必须提供整数 order_id")
        return state
    plan_data = build_plan_for_route(route, order_id, a, b)
    attach_plan(state, plan_data)
    start_task(state)
    if route == "conditional":
        while (
                state["status"] == "running" and state["current_step_index"] < len(state["plan"])):
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
    else:
        execute_plan(state)

    if state["status"] != "running":
        return state
    if any(step["status"] not in ("completed", "skipped") for step in state["plan"]):
        fail_task(state, "仍存在未结束的计划步骤")
        return state
    if route == "order_only":
        result = state["plan"][0]["result"]
        answer = (
            f"订单{result['order_id']}状态为"
            f"{result['status']}。"
        )

    elif route == "multiply_only":
        result = state["plan"][0]["result"]
        answer = f"乘积为{result['product']}。"

    else:
        order_result = state["plan"][0]["result"]
        multiply_step = state["plan"][1]

        if multiply_step["status"] == "skipped":
            answer = (
                f"订单状态为{order_result['status']}，"
                "未满足计算条件，已跳过乘法。"
            )
        else:
            product = multiply_step["result"]["product"]
            answer = (
                f"订单状态为{order_result['status']}，"
                f"乘积为{product}。"
            )

    finish_task(state, answer)

    return state
