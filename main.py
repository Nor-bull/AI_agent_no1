
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def calculate(expression: str):
    return {"expression": expression, "result": f"(LLM simuluje výpočet {expression})"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Vyhodnotí matematický výraz a vrátí výsledek.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    }
]

available_functions = {
    "calculate": calculate
}


def run_with_tool(messages, model="gpt-4.1-nano"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="required"
    )

    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            expression = args["expression"]
            tool_id = tool_call.id

            function_response = available_functions[tool_call.function.name](expression)

            follow_up_messages = messages + [
                assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(function_response)
                }
            ]

            final_response = client.chat.completions.create(
                model=model,
                messages=follow_up_messages,
                tools=tools
            )

            print("=== Odpoveď LLM po použití toolu ===")
            print(final_response.choices[0].message.content)

    else:
        print("=== Model odpovedal bez toolu ===")
        print(assistant_message.content)


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Kolik je (x + y) * z?"}
]

run_with_tool(messages)