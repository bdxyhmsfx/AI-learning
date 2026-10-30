from agent_state import AgentState, record_tool_result, fail_task
from recovery import execute_with_retry, RetryExhaustedError, ToolAttemptBudgetExceeded


def execute_next_step(state: AgentState) -> bool:
    if state["status"] != "running":
        raise ValueError("只有 running 任务才能执行计划")
    plan = state["plan"]
    if not plan:
        raise ValueError("当前任务没有可执行计划")
    index = state["current_step_index"]
    if index >= len(plan):
        state["current_step"] = "awaiting_final_answer"
        return False
    step = plan[index]
    if step["status"] != "pending":
        raise ValueError("当前步骤不是 pending 状态")
    for dep_id in step["depends_on"]:
        dependency = plan[dep_id - 1]
        if dependency["status"] != "completed":
            step["status"] = "failed"
            fail_task(state, f"步骤 {step['id']} 的前置步骤 {dep_id} 未完成")
            return False
    step["status"] = "running"
    state["current_step"] = f"execute_step_{step['id']}"
    print(f"\n开始执行步骤 {step['id']}: {step['description']}")
    try:
        result = execute_with_retry(state, step)
    except Exception as e:
        known_errors = (
            ValueError,
            TypeError,
            RetryExhaustedError,
            ToolAttemptBudgetExceeded
        )
        if isinstance(e, known_errors):
            error_message = str(e)
        else:
            error_message = "工具执行出现未预期的内部错误"
        if step.get("attempts", 0) > 0:
            record_tool_result(state, step["tool_name"], {"error": error_message})
        step["status"] = "failed"
        step["error"] = error_message
        fail_task(state, f"步骤 {step['id']} 执行失败：{error_message}")

        return False

    step["result"] = result
    record_tool_result(state, step["tool_name"], result)
    step["status"] = "completed"
    state["current_step_index"] += 1
    print("执行结果：", result)
    print("当前步骤状态：", step["status"])
    if state["current_step_index"] == len(plan):
        state["current_step"] = "awaiting_final_answer"
    return True


def execute_plan(state: AgentState) -> AgentState:
    if state["status"] != "running":
        raise ValueError("任务未处于 running 状态")
    if not state["plan"]:
        raise ValueError("当前任务没有可执行计划")
    while (state["status"] == "running" and state["current_step_index"] < len(state["plan"])):
        execute_next_step(state)
    return state
