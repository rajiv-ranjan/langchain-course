from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 5

## Observation: When uisng qwen3.5:27b, the agent give a list of tools to use in each iteration.
## Observation: When uisng gpt-oss:latest, the agent gives only one tool to use in each iteration. As I have set the max_iterations to 5, the agent will iterate 5 times and then max out with error message "ERROR: Max iterations reached without a final answer".
## Switch between the two models to see the difference.
# MODEL = "qwen3.5:27b"
MODEL = "gpt-oss:latest"

MODEL_PROVIDER = "ollama"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1000.00, "headphones": 500.00, "keyboard": 100.00}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

@traceable(project="project3-react-under-the-hood")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    model = init_chat_model(model=MODEL, model_provider=MODEL_PROVIDER,temperature=0.0)
    model_with_tools = model.bind_tools(tools)


    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = model_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            observation = tool_to_use.invoke(tool_args)

            print(f"  [Tool Result] {observation}")

            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    # run_agent("What is the price of a laptop with a gold discount?")
    # run_agent("What comes after alphabet A?")
    # run_agent("What is the price of a laptop with all gold, silver and bronze discounts?")
    run_agent("What is the price of a laptop and headphones with all gold, silver and bronze discounts?")