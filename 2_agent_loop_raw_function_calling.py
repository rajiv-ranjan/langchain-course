from typing_extensions import runtime
from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 5

## Observation: When uisng qwen3.5:27b, the agent give a list of tools to use in each iteration.
## Observation: When uisng gpt-oss:latest, the agent gives only one tool to use in each iteration. As I have set the max_iterations to 5, the agent will iterate 5 times and then max out with error message "ERROR: Max iterations reached without a final answer".
## Switch between the two models to see the difference.
MODEL = "qwen3.5:27b"
# MODEL = "gpt-oss:latest"

MODEL_PROVIDER = "ollama"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.

    Args:
        product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.

    Returns:
        The price of the product, or 0 if not found.
    """
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1000.00, "headphones": 500.00, "keyboard": 100.00}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_product_price",
    #         "description": "Look up the price of a product in the catalog.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "product": {
    #                     "type": "string",
    #                     "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
    #                 },
    #             },
    #             "required": ["product"],
    #         },
    #     },
    # },
    get_product_price,
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    # tools_dict = {tool.name: tool for tool in tools_for_llm}
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    # response = ollama_chat_traced(messages=messages)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            # tool_call_id = tool_call.id

            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            observation = tool_to_use(**tool_args)

            print(f"  [Tool Result] {observation}")

            messages.append(
                {
                    "role": "tool",
                    "content": str(observation),
                    # "tool_call_id": tool_call.id,
                }
            )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    run_agent("What is the price of a laptop with a gold discount?")
    # run_agent("What comes after alphabet A?")
    # run_agent("What is the price of a laptop with all gold, silver and bronze discounts?")
    # run_agent(
    #     "What is the price of a laptop and headphones with all gold, silver and bronze discounts?"
    # )
