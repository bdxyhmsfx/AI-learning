from agent_state import AgentState, create_agent_state, start_task, finish_task, fail_task
from planner import generate_plan, attach_plan
from executor import execute_next_step
from observer import evaluate_step_condition


def build_final_answer(state: AgentState) -> str:
    parts = []
    for step in state["plan"]:
        if step["status"] == "skipped":
            parts.append(
                f"步骤{step['id']}已跳过："
                f"{step.get('skip_reason', '条件不满足')}"
            )
            continue
        if step["status"] != "completed":
            continue
        result = step["result"]
        if step["tool_name"] == "get_order_status":
            parts.append(
                f"订单{result['order_id']}状态为"
                f"{result['status']}。"
            )
        elif step["tool_name"] == "multiply":
            parts.append(
                f"步骤{step['id']}的乘积为"
                f"{result['product']}。"
            )

    return "\n".join(parts)


def run_agent_workflow(query: str, plan_data: dict | None = None) -> AgentState:
    state = create_agent_state(query)
    try:
        if plan_data is None:
            plan_data = generate_plan(query)
        if plan_data == {"steps": []}:
            fail_task(state, "当前工具无法完成该任务")
            return state
        attach_plan(state, plan_data)
    except (ValueError, TypeError) as e:
        fail_task(state, f"计划校验失败：{e}")
        return state

    except Exception:
        fail_task(state, "计划生成过程中出现外部服务或内部错误")
        return state
    start_task(state)
    while (state["status"] == "running" and state["current_step_index"] < len(state["plan"])):
        index = state["current_step_index"]
        step = state["plan"][index]
        skipped_dependency = next(
            (
                dep_id
                for dep_id in step["depends_on"]
                if state["plan"][dep_id - 1]["status"] == "skipped"
            ),
            None
        )

        if skipped_dependency is not None:
            step["status"] = "skipped"
            step["skip_reason"] = (
                f"前置步骤{skipped_dependency}已跳过"
            )

            state["current_step_index"] += 1
            continue

        try:
            can_execute = evaluate_step_condition(state, step)
        except (ValueError, TypeError, KeyError) as e:
            step["status"] = "failed"
            fail_task(state, f"步骤条件检查失败：{e}")
            break
        if not can_execute:
            step["status"] = "skipped"
            step["skip_reason"] = "业务执行条件不满足"
            state["current_step_index"] += 1
            continue
        success = execute_next_step(state)
        if not success:
            break

    if state["status"] != "running":
        return state
    if any(step["status"] not in ("completed", "skipped")for step in state["plan"]):
        fail_task(state, "计划仍存在未完成步骤")
        return state
    try:
        answer = build_final_answer(state)
        finish_task(state, answer)

    except (ValueError, TypeError, KeyError) as e:
        fail_task(state, f"最终结果生成失败：{e}")

    return state
