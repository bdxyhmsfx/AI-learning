import os
import json
from openai import OpenAI
from tool_schemas import multiply_tool, get_order_status_tool
from tool_dispatcher import execute_tool

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL_NAME = "deepseek/deepseek-v3.2"


def to_chat_tool(schema):
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
            "strict": schema["strict"]
        }
    }


tools = [to_chat_tool(multiply_tool), to_chat_tool(get_order_status_tool)]


def run_tool_conversation(query, max_rounds=5, allow_parallel=True):
    messages = [{
        "role": "user", "content": query
    }]
    for round_index in range(1, max_rounds + 1):
        print(f"\n========== 第 {round_index} 轮 ==========")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=allow_parallel
        )
        message = response.choices[0].message
        if not message.tool_calls:
            print("最终的回答：", message.content)
            return message.content
        messages.append(message.model_dump(exclude_none=True))
        print("本轮工具调用数量：", len(message.tool_calls))
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments_json = tool_call.function.arguments
            print("\n工具名称：", tool_name)
            print("原始参数：", arguments_json)
            try:
                arguments = json.loads(arguments_json)
                result = execute_tool(tool_name, arguments)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                result = {"error": str(e)}
            print("工具结果：", result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
    raise RuntimeError(f"达到最大模型请求轮数 {max_rounds}，仍未获得最终回答")
if __name__ == "__main__":
    query = (
        "请解释 Tool Calling 与 RAG 的主要区别。"
    )

    run_tool_conversation(query)