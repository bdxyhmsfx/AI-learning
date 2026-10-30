from agent_state import AgentState
from agent_learning.tool_calling.tool_dispatcher import execute_tool


class RetryExhaustedError(RuntimeError):
    """可重试错误连续发生，尝试次数已耗尽"""
    pass


class ToolAttemptBudgetExceeded(RuntimeError):
    """整个任务的工具尝试次数超过预算"""
    pass


def execute_with_retry(
        state: AgentState,
        step: dict,
        max_attempts: int = 2,
        max_total_attempts: int = 5
) -> dict:
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts 必须是正整数")
    if type(max_total_attempts) is not int or max_total_attempts < 1:
        raise ValueError("max_total_attempts 必须是正整数")
    for attempt in range(1, max_attempts + 1):
        if state["tool_attempt_count"] >= max_total_attempts:
            raise ToolAttemptBudgetExceeded("任务的工具尝试次数预算已耗尽")
        state["tool_attempt_count"] += 1
        step["attempts"] = attempt
        try:
            result = execute_tool(step["tool_name"], step["arguments"])
            return result
        except (TimeoutError, ConnectionError) as e:
            if attempt == max_attempts:
                raise RetryExhaustedError(
                    f"步骤 {step['id']} 尝试 {attempt} 次后仍失败"
                ) from e
            print(
                f"步骤 {step['id']} 第 {attempt} 次执行失败，"
                "准备重试"
            )
    raise RuntimeError("执行流程异常")
