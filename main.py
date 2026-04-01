import re
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from tavily import TavilyClient

load_dotenv()


class Source(BaseModel):
    """Schema for the source of the search results"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to answer the query"
    )
    answer: str = Field(description="The answer to the query")


tavily_client = TavilyClient()


@tool
def search_capital(full_query: str) -> str:
    """
    Tool should be used only to find the capital of a country

    Args:
        full_query: The full query to search as string

    Returns:
        The capital of the country as string
    """
    print(f"Query: {full_query}")
    results = tavily_client.search(query=full_query)
    print(f"Results: {results}")
    return results


tools = [search_capital, TavilySearch()]


def select_provider() -> str:
    """Prompt user to select LLM provider."""
    print("\nSelect LLM Provider:")
    print("1. OpenAI (supports structured output)")
    print("2. Ollama (local models)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return "openai"
        elif choice == "2":
            return "ollama"
        print("Invalid choice. Please enter 1 or 2.")


OLLAMA_MODELS = {
    "1": {"name": "gpt-oss:latest", "structured": True},   # Tools: YES → ToolStrategy works
    "2": {"name": "qwen3.5:27b", "structured": True},      # Tools: YES → ToolStrategy works
    "3": {"name": "qwen3:8b", "structured": True},         # Tools: YES → ToolStrategy works
    "4": {"name": "llama3.1:8b", "structured": True},      # Tools: YES → ToolStrategy works
    "5": {"name": "gemma3:12b", "structured": False},      # Tools: LIMITED → ToolStrategy unreliable
}

STRUCTURED_OUTPUT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

IMPORTANT: When you have gathered enough information to answer the user's question, you MUST use the 'AgentResponse' tool to provide your final answer. Do NOT respond with plain text.

The AgentResponse tool requires:
- answer: Your complete answer to the user's question
- sources: A list of source URLs used to form your answer (each with a 'url' field)

Always call the AgentResponse tool for your final response. Never skip this step.
Limit yourself to a maximum of 3 tool calls before providing your final answer."""

MAX_ITERATIONS = 10  # Maximum number of agent steps/tool calls
RECURSION_LIMIT = 25  # Maximum recursion depth for the agent graph


def create_llm_and_agent(provider: str):
    """Create LLM and agent based on selected provider."""
    if provider == "openai":
        print("\nUsing OpenAI with structured output...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        agent = create_agent(
            model=llm,
            tools=tools,
            response_format=AgentResponse,
        )
        return agent, True

    print("\nSelect Ollama model (all use ToolStrategy for structured output):")
    print("1. gpt-oss:latest    (ToolStrategy: YES)")
    print("2. qwen3.5:27b       (ToolStrategy: YES) - Best for agents")
    print("3. qwen3:8b          (ToolStrategy: YES)")
    print("4. llama3.1:8b       (ToolStrategy: YES)")
    print("5. gemma3:12b        (ToolStrategy: NO - limited tool support)")
    print("6. Custom (enter name)")

    model_choice = input("Enter choice (1-6): ").strip()

    if model_choice in OLLAMA_MODELS:
        model_info = OLLAMA_MODELS[model_choice]
        model_name = model_info["name"]
        supports_structured = model_info["structured"]
    else:
        model_name = input("Enter model name: ").strip()
        use_structured = input("Enable structured output? (y/n): ").strip().lower()
        supports_structured = use_structured == "y"

    llm = ChatOllama(model=model_name, temperature=0.0)

    if supports_structured:
        print(f"\nUsing Ollama ({model_name}) WITH ToolStrategy structured output...")
        agent = create_agent(
            model=llm,
            tools=tools,
            response_format=ToolStrategy(schema=AgentResponse),
        )
        return agent, True
    else:
        print(f"\nUsing Ollama ({model_name}) WITHOUT structured output...")
        agent = create_agent(model=llm, tools=tools)
        return agent, False


console = Console()


def extract_sources_from_content(content: str) -> list:
    """Extract URLs from markdown-style content."""
    urls = re.findall(r'https?://[^\s\)>\]]+', content)
    return [{"url": url.rstrip('.,;:')} for url in urls]


def format_response(result, supports_structured: bool, provider: str):
    """Format and display the agent response."""
    structured_response = result.get("structured_response")
    
    # Case 1: ToolStrategy worked - we have structured_response
    if supports_structured and structured_response:
        if isinstance(structured_response, AgentResponse):
            data = structured_response.model_dump()
        elif isinstance(structured_response, dict):
            data = structured_response
        else:
            data = {"response": str(structured_response)}
        
        title = "[bold green]STRUCTURED RESPONSE (ToolStrategy)[/bold green]" if provider == "ollama" else "[bold blue]STRUCTURED RESPONSE[/bold blue]"
        border = "green" if provider == "ollama" else "blue"
        console.print(Panel(JSON.from_data(data), title=title, border_style=border))
        return

    messages = result.get("messages", [])
    if not messages:
        print("No response received.")
        return

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Case 2: ToolStrategy didn't work but we have content - manually structure it
    if supports_structured and provider == "ollama" and content:
        console.print("[yellow]Note: ToolStrategy fallback - model responded with content instead of tool call[/yellow]")
        sources = extract_sources_from_content(content)
        fallback_data = {
            "answer": content,
            "sources": sources,
            "_note": "Manually extracted (ToolStrategy fallback)"
        }
        console.print(Panel(JSON.from_data(fallback_data), title="[bold yellow]RESPONSE (ToolStrategy Fallback)[/bold yellow]", border_style="yellow"))
        return

    # Case 3: OpenAI structured or plain response
    if supports_structured and provider == "openai":
        console.print(Panel(content, title="[bold blue]STRUCTURED RESPONSE[/bold blue]", border_style="blue"))
    else:
        console.print(Panel(content, title="[bold cyan]RESPONSE[/bold cyan]", border_style="cyan"))


def main():
    provider = select_provider()
    agent, supports_structured = create_llm_and_agent(provider)
    
    print("\nAgent ready! Type 'exit' to quit, 'switch' to change provider.\n")
    
    while True:
        user_input = input("Enter a query: ").strip()
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "switch":
            provider = select_provider()
            agent, supports_structured = create_llm_and_agent(provider)
            print("\nProvider switched! Ready for queries.\n")
            continue
        
        if not user_input:
            continue

        enhanced_query = f"{user_input}. Include the source URLs for each result."

        try:
            if supports_structured and provider == "ollama":
                messages = [
                    SystemMessage(content=STRUCTURED_OUTPUT_SYSTEM_PROMPT),
                    HumanMessage(content=enhanced_query),
                ]
            else:
                messages = [HumanMessage(content=enhanced_query)]
            
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": RECURSION_LIMIT},
            )
            format_response(result, supports_structured, provider)
        except Exception as e:
            error_msg = str(e)
            if "recursion" in error_msg.lower() or "limit" in error_msg.lower():
                console.print(f"[bold yellow]Agent stopped:[/bold yellow] Maximum iterations ({RECURSION_LIMIT}) reached. The agent may be in a loop.\n")
            else:
                console.print(f"[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
