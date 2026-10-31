import os
import json
from openai import OpenAI
from typing import Any
from agent_state import AgentState

ALLOWED_TOOLS = {"multiply", "get_order_status"}
MAX_PLAN_STEPS = 5
MODEL_NAME = "deepseek/deepseek-v3.2"
PLANNER_PROMPT = """
你是一个任务规划器 Planner。

你的职责是将用户任务拆解为可执行的工具步骤。

当前允许的工具：

1. get_order_status
   参数：{"order_id": 整数}

2. multiply
   参数：{"a": 数字, "b": 数字}

规则：
1. 只能使用上述工具，不能编造新工具。
2. 步骤 ID 从 1 开始连续编号。
3. depends_on 只能引用之前的步骤 ID。
4. 最多生成 5 个步骤。
5. 仅规划工具执行，不需要生成最终回答。
6. 不能完成的任务返回 {"steps": []}。
7. 只返回 JSON 对象，不要 Markdown 代码块。
8. 必须根据用户描述决定工具的执行顺序，不能固定先查询订单再计算乘法。
9. depends_on 只表达步骤执行先后关系。
10. 如果用户明确要求“只有订单达到某状态才执行后续步骤”，
    必须为对应的后续步骤增加 run_if：
    {"step_id": 前置订单查询步骤ID, "status_equals": "要求的状态"}
11. run_if 中的 step_id 必须同时出现在 depends_on 中。
12. 没有业务条件的步骤，不要添加 run_if。
13. 当前 run_if 只支持检查 get_order_status 返回的订单状态。

返回格式：
{
    "steps": [
        {
            "id": 1,
            "description": "步骤描述",
            "tool_name": "工具名称",
            "arguments": {},
            "depends_on": []
        }
    ]
}
"""


def validate_plan(plan_data: Any) -> list[dict[str, Any]]:
    if not isinstance(plan_data, dict):
        raise TypeError("Plan 必须是字典")
    if set(plan_data) != {"steps"}:
        raise ValueError("Plan 必须包含 steps 字段")
    steps = plan_data["steps"]
    if not isinstance(steps, list):
        raise TypeError("steps 必须是列表")
    if not 1 <= len(steps) <= MAX_PLAN_STEPS:
        raise ValueError("计划步骤数量不符合要求")
    validated_steps = []
    required_fields = {"id", "description", "tool_name", "arguments", "depends_on"}
    allowed_fields = required_fields | {"run_if"}
    for expected_id, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise TypeError("每个步骤必须是字典")
        fields = set(step)
        if not required_fields.issubset(fields):
            raise ValueError("步骤缺少必要字段")
        if not fields.issubset(allowed_fields):
            raise ValueError("步骤存在不允许的字段")
        if type(step["id"]) is not int:
            raise TypeError("步骤 ID 必须是整数")
        if step["id"] != expected_id:
            raise ValueError("步骤 ID 必须从 1 开始连续排列")
        if not isinstance(step["description"], str):
            raise TypeError("description 必须是字符串")
        if not step["description"].strip():
            raise ValueError("步骤描述不能为空")
        if step["tool_name"] not in ALLOWED_TOOLS:
            raise ValueError(f"不允许规划的工具：{step['tool_name']}")
        if not isinstance(step["arguments"], dict):
            raise TypeError("arguments 必须是字典")
        dependencies = step["depends_on"]
        if not isinstance(dependencies, list):
            raise TypeError("depends_on 必须是列表")
        for dep in dependencies:
            if type(dep) is not int or not 1 <= dep < expected_id:
                raise ValueError("步骤只能依赖前面的步骤")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("依赖关系不能重复")
        if "run_if" in step:
            guard = step["run_if"]
            if not isinstance(guard, dict):
                raise TypeError("run_if 必须是字典")
            if set(guard) != {"step_id", "status_equals"}:
                raise ValueError("run_if 字段不符合要求")
            source_id = guard["step_id"]
            if type(source_id) is not int:
                raise TypeError("run_if.step_id 必须是整数")
            if not 1 <= source_id < expected_id:
                raise ValueError("条件只能引用之前的步骤")
            if source_id not in dependencies:
                raise ValueError("条件来源必须同时出现在 depends_on 中")
            source_step = steps[source_id - 1]
            if source_step["tool_name"] != "get_order_status":
                raise ValueError("当前只支持订单查询结果作为执行条件")
            expected_status = guard["status_equals"]

            if not isinstance(expected_status, str) or not expected_status.strip():
                raise ValueError("status_equals 必须是非空字符串")
        validated_steps.append({**step, "status": "pending"})
    return validated_steps


def attach_plan(state: AgentState, plan_data: Any) -> None:
    if state["status"] != "pending":
        raise ValueError("只能为 pending 任务设置计划")
    if state["plan"]:
        raise ValueError("当前任务已经存在计划")
    validated_plan = validate_plan(plan_data)
    state["plan"] = validated_plan
    state["current_step_index"] = 0


def generate_plan(query: str) -> dict:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("模型没有返回计划内容")

    try:
        plan_data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError("模型没有返回合法的 JSON 计划") from e

    return plan_data
