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
query = "请简单解释数据库索引是什么。"
messages = [{"role": "user", "content": query}]
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
message = response.choices[0].message
if not message.tool_calls:
    print("模型未请求工具")
    print("直接回答：", message.content)
else:
    messages.append(message.model_dump(exclude_none=True))  # 转为python字典，并删除None的字段
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
        print("工具执行结果：", result)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                result,
                ensure_ascii=False
            )
        })

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="none"
    )
    answer = final_response.choices[0].message.content
    print("\n最终回答：", answer)