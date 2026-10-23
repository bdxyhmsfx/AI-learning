import os
import json
from openai import OpenAI
from tool_schemas import multiply_tool, get_order_status_tool

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
query = "请使用乘法工具计算 12.5 和 8 的乘积。"
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": query
        }
    ],
    tools=tools,
    tool_choice="auto"
)
choice = response.choices[0]
message = choice.message
print("Finish Reason:", choice.finish_reason)
if message.tool_calls:
    for tool_call in message.tool_calls:
        print("\n===== Tool Call =====")
        print("Call ID:", tool_call.id)
        print("Tool Name:", tool_call.function.name)
        print("Raw Arguments:", tool_call.function.arguments)
        arguments = json.loads(tool_call.function.arguments)
        print("Parsed Arguments:", arguments)
else:
    print("\n模型没有请求工具。")
    print("Model Answer:", message.content)
