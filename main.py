
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def calculate(expression: str):
    """
    Tento Python funkcia tu NEVYKONÁVA skutočný výpočet,
    iba vracia expression späť do LLM ako 'tool result'.
    """
    # Tu môžeš simulovať výsledok alebo vrátiť string "(simulovaný výpočet)"
    return {"expression": expression, "result": f"(LLM simuluje výpočet {expression})"}


tools = [
    {
        "type": "function",
        "name": "calculate",
        "description": "Vyhodnotí matematický výraz a vrátí výsledek.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]

available_functions = {
    "calculate": calculate
}


def run_with_tool(messages, model="gpt-4.1-mini"):
    response = client.responses.create(
        model=model,
        input=messages[-1]["content"],
        tools=tools,
        tool_choice="required"  # prinúti model použiť tool
    )

    tool_called = False
    for item in response.output:
        if item.type == "tool_call":
            tool_called = True
            args = json.loads(item.arguments)
            expression = args["expression"]
            tool_id = item.id

            function_response = available_functions[item.name](expression)

            final_response = client.responses.create(
                model=model,
                input=[
                    {"role": "user", "content": messages[-1]["content"]},
                    {
                        "type": "tool_result",
                        "tool_call_id": tool_id,
                        "content": json.dumps(function_response)
                    }
                ],
                tools=tools
            )

            print("=== Odpoveď LLM po použití toolu ===")
            print(final_response.output_text)

    if not tool_called:
        print("=== Model odpovedal bez toolu ===")
        print(response.output_text)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Kolik je (12 + 8) * 3?"}
]

run_with_tool(messages)
