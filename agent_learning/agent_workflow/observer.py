from agent_state import AgentState
def observe_order_result(state: AgentState,expected_status: str = "已完成") -> dict | None:
    if state["status"] != "running":
        raise ValueError("任务未处于 running 状态")
    last_index = state["current_step_index"] - 1
    if last_index < 0:
        return None
    step = state["plan"][last_index]
    if step["status"] != "completed":
        return None
    if step["tool_name"] != "get_order_status":
        return None
    result = step["result"]
    actual_status = result["status"]
    observation = {
        "step_id":step["id"],
        "actual_status":actual_status,
        "expected_status":expected_status,
        "condition_met":actual_status == expected_status,
    }
    state["observations"].append(observation)
    return observation

def replan_after_observation(state: AgentState,observation: dict) -> int:
    if state["status"] != "running":
        raise ValueError("任务未处于 running 状态")
    if observation["condition_met"]:
        return 0
    source_step_id = observation["step_id"]
    skipped_ids = set()
    remaining_steps = state["plan"][state["current_step_index"]:]
    for step in remaining_steps:
        if step["status"] != "pending":
            continue
        dependencies = step["depends_on"]
        should_skip = (
            source_step_id in dependencies
            or any(dep in skipped_ids for dep in dependencies)
        )
        if should_skip:
            step["status"] = "skipped"
            step["skip_reason"] = (f"步骤{source_step_id}的业务条件未满足")
            skipped_ids.add(step["id"])
    print("重新规划：跳过步骤", sorted(skipped_ids))
    return len(skipped_ids)

def evaluate_step_condition(state: AgentState,step:dict) -> bool:
    guard = step.get("run_if")
    if guard is None:
        return True
    source_id = guard["step_id"]
    source_step = state["plan"][source_id - 1]
    if source_step["status"] != "completed":
        raise ValueError("条件来源步骤尚未完成")
    result = source_step["result"]
    if not isinstance(result, dict) or "status" not in result:
        raise ValueError("条件来源缺少订单状态")
    actual_status = result["status"]
    expected_status = guard["status_equals"]
    condition_met = actual_status == expected_status
    observation = {
        "step_id": source_id,
        "guarded_step_id": step["id"],
        "actual_status": actual_status,
        "expected_status": expected_status,
        "condition_met": condition_met
    }
    state["observations"].append(observation)

    return condition_met
