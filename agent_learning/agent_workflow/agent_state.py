from typing import TypedDict, Literal, Any


class AgentState(TypedDict):
    user_query: str
    messages: list[dict[str, Any]]
    status: Literal[
        "pending",
        "running",
        "completed",
        "failed"
    ]
    current_step: str | None
    tool_results: list[dict[str, Any]]
    round_count: int
    tool_call_count: int
    final_answer: str | None
    error: str | None


def create_agent_state(query: str) -> AgentState:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    query = query.strip()
    state: AgentState = {
        "user_query": query,
        "messages": [{"role": "user", "content": query}],
        "status": "pending",
        "current_step": None,
        "tool_results": [],
        "round_count": 0,
        "tool_call_count": 0,
        "final_answer": None,
        "error": None
    }
    return state


def start_task(state: AgentState) -> None:
    if state["status"] != "pending":
        raise ValueError("只有 pending 任务可以开始")
    state["status"] = "running"
    state["current_step"] = "model_decision"


def record_round(state: AgentState) -> None:
    if state["status"] != "running":
        raise ValueError("任务未处于 running 状态")
    state["round_count"] += 1


def record_tool_result(state: AgentState, tool_name: str, result: dict) -> None:
    if state["status"] != "running":
        raise ValueError("当前任务不允许记录工具结果")
    state["tool_results"].append({"tool": tool_name, "result": result})
    state["tool_call_count"] += 1


def finish_task(state: AgentState, answer: str) -> None:
    if state["status"] != "running":
        raise ValueError("只有 running 任务可以完成")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("最终回答不能为空")
    state["final_answer"] = answer.strip()
    state["status"] = "completed"
    state["current_step"] = None


def fail_task(state: AgentState, error: str) -> None:
    if state["status"] not in ("pending", "running"):
        raise ValueError("任务已经结束，不能重复标记失败")
    state["error"] = str(error)
    state["status"] = "failed"
    state["current_step"] = None
