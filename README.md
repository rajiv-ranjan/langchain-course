# LangChain Agent with Structured Output

A LangChain-based agent that supports both OpenAI and Ollama models with structured output capabilities using Pydantic schemas.

## Features

- **Multi-provider support**: OpenAI (GPT-4o-mini) and Ollama (local models)
- **Structured output**: Uses `ProviderStrategy` for OpenAI and `ToolStrategy` for Ollama
- **Web search tools**: Tavily Search integration
- **Pretty output**: Rich console formatting with JSON highlighting
- **Fallback handling**: Graceful degradation when ToolStrategy fails

## Architecture

### Component Diagram

```mermaid
graph TB
    subgraph User Interface
        UI[main.py]
        Console[Rich Console]
    end

    subgraph Provider Selection
        PS[select_provider]
        PS --> OpenAI
        PS --> Ollama
    end

    subgraph LLM Providers
        OpenAI[ChatOpenAI<br/>gpt-4o-mini]
        Ollama[ChatOllama<br/>qwen3.5/llama3.1/etc]
    end

    subgraph Agent Factory
        CA[create_agent]
        CA --> |OpenAI| ProvStrat[ProviderStrategy<br/>Native structured output]
        CA --> |Ollama| ToolStrat[ToolStrategy<br/>Tool-based structured output]
    end

    subgraph Structured Output Schema
        AR[AgentResponse<br/>Pydantic Model]
        AR --> Answer[answer: str]
        AR --> Sources[sources: List&lt;Source&gt;]
        Sources --> URL[url: str]
    end

    subgraph Tools
        T1[search_capital<br/>Custom Tool]
        T2[TavilySearch<br/>Web Search]
        TC[TavilyClient]
    end

    subgraph Response Formatting
        FR[format_response]
        FR --> Structured[Structured Response<br/>Green/Blue Panel]
        FR --> Fallback[Fallback Response<br/>Yellow Panel]
        FR --> Plain[Plain Response<br/>Cyan Panel]
    end

    UI --> PS
    PS --> CA
    CA --> AR
    CA --> T1
    CA --> T2
    T1 --> TC
    OpenAI --> ProvStrat
    Ollama --> ToolStrat
    ProvStrat --> AR
    ToolStrat --> AR
    UI --> FR
    FR --> Console

    style OpenAI fill:#74aa9c,color:#fff
    style Ollama fill:#1a1a2e,color:#fff
    style ProvStrat fill:#4a90d9,color:#fff
    style ToolStrat fill:#2ecc71,color:#fff
    style AR fill:#9b59b6,color:#fff
```

### Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        ENV[.env<br/>API Keys]
        Query[User Query]
    end

    subgraph LLM Layer
        Agent[LangChain Agent]
        LLM[LLM Model<br/>OpenAI / Ollama]
    end

    subgraph Tools Layer
        direction TB
        subgraph Custom Tools
            SC[search_capital<br/>Capital Finder Tool]
        end
        subgraph LangChain Tools
            TS[TavilySearch<br/>Web Search Tool]
        end
        subgraph External Clients
            TC[TavilyClient<br/>Direct API Client]
        end
    end

    subgraph External APIs
        TavilyAPI[(Tavily API)]
        Web[(Web Search Results)]
    end

    subgraph Output
        SR[structured_response<br/>AgentResponse]
        MSG[messages<br/>AIMessage]
        Console[Rich Console<br/>Pretty Print]
    end

    ENV --> |OPENAI_API_KEY| LLM
    ENV --> |TAVILY_API_KEY| TC
    ENV --> |TAVILY_API_KEY| TS
    Query --> Agent
    
    Agent <--> LLM
    Agent --> |tool_call| SC
    Agent --> |tool_call| TS
    
    SC --> TC
    TC --> |HTTP| TavilyAPI
    TS --> |HTTP| TavilyAPI
    TavilyAPI --> Web
    Web --> |JSON| TavilyAPI
    TavilyAPI --> |results| TC
    TavilyAPI --> |results| TS
    TC --> |tool_response| SC
    SC --> |tool_response| Agent
    TS --> |tool_response| Agent
    
    Agent --> SR
    Agent --> MSG
    SR --> Console
    MSG --> Console

    style SC fill:#e74c3c,color:#fff
    style TS fill:#3498db,color:#fff
    style TC fill:#9b59b6,color:#fff
    style Agent fill:#2ecc71,color:#fff
    style LLM fill:#f39c12,color:#fff
    style TavilyAPI fill:#1abc9c,color:#fff
```

## Sequence Diagram

### Main Flow - OpenAI with ProviderStrategy

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as main()
    participant SP as select_provider()
    participant CA as create_agent()
    participant OpenAI as ChatOpenAI
    participant Agent as LangChain Agent
    participant Tools as TavilySearch
    participant Web as Tavily API
    participant FR as format_response()
    participant C as Console

    U->>M: Run application
    M->>SP: Select provider
    SP-->>M: "openai"
    
    M->>CA: create_agent(model, tools, response_format)
    CA->>OpenAI: Initialize GPT-4o-mini
    CA-->>M: agent, supports_structured=True

    loop Query Loop
        U->>M: Enter query
        M->>Agent: invoke(messages)
        
        Agent->>OpenAI: Process query
        OpenAI-->>Agent: Decide to use tool
        
        Agent->>Tools: search_capital(query)
        Tools->>Web: HTTP Request
        Web-->>Tools: Search results
        Tools-->>Agent: Tool response
        
        Agent->>OpenAI: Process tool results
        OpenAI-->>Agent: structured_response (AgentResponse)
        
        Agent-->>M: result with structured_response
        M->>FR: format_response(result)
        FR->>C: Display Blue Panel
        C-->>U: Formatted output
    end
```

### Ollama with ToolStrategy Flow

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as main()
    participant CA as create_agent()
    participant Ollama as ChatOllama
    participant Agent as LangChain Agent
    participant Tools as Search Tools
    participant TS as ToolStrategy
    participant FR as format_response()
    participant C as Console

    M->>CA: create_agent(model, tools, ToolStrategy)
    CA->>Ollama: Initialize (qwen3.5:27b)
    CA->>TS: Setup AgentResponse as tool
    CA-->>M: agent, supports_structured=True

    U->>M: Enter query
    
    Note over M: Add SystemMessage with<br/>structured output instructions
    
    M->>Agent: invoke([SystemMessage, HumanMessage])
    
    Agent->>Ollama: Process with system prompt
    Ollama-->>Agent: Tool call: search_capital
    
    Agent->>Tools: Execute search_capital
    Tools-->>Agent: Search results
    
    Agent->>Ollama: Process results
    
    alt ToolStrategy Success
        Ollama-->>Agent: Tool call: AgentResponse
        Agent->>TS: Extract structured data
        TS-->>Agent: AgentResponse object
        Agent-->>M: result["structured_response"]
        M->>FR: format_response()
        FR->>C: Green Panel (ToolStrategy)
    else ToolStrategy Fallback
        Ollama-->>Agent: Plain text content
        Agent-->>M: result["messages"]
        M->>FR: format_response()
        Note over FR: Extract URLs with regex
        FR->>C: Yellow Panel (Fallback)
    end
    
    C-->>U: Formatted output
```

### Error Handling Flow

```mermaid
sequenceDiagram
    participant U as User
    participant M as main()
    participant Agent as LangChain Agent
    participant C as Console

    U->>M: Enter query
    M->>Agent: invoke(messages, recursion_limit=25)
    
    alt Normal Execution
        Agent-->>M: result
        M->>C: Display response
    else Recursion Limit Exceeded
        Agent--xM: RecursionError
        M->>C: Yellow warning message
    else Other Error
        Agent--xM: Exception
        M->>C: Red error message
    end
    
    C-->>U: Output/Error
```

## Structured Output Strategies

```mermaid
graph LR
    subgraph Strategy Selection
        RF[response_format parameter]
    end

    subgraph OpenAI Path
        PS[ProviderStrategy]
        PS --> |Native API| NSO[Native Structured Output]
        NSO --> |Guaranteed| Schema1[AgentResponse Schema]
    end

    subgraph Ollama Path
        TS[ToolStrategy]
        TS --> |Creates hidden tool| RT[Response Tool]
        RT --> |Model calls tool| Schema2[AgentResponse Schema]
        RT -.-> |May skip tool| FB[Fallback Parser]
    end

    RF --> |OpenAI| PS
    RF --> |Ollama| TS

    style PS fill:#4a90d9,color:#fff
    style TS fill:#2ecc71,color:#fff
    style FB fill:#f39c12,color:#fff
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RECURSION_LIMIT` | 25 | Maximum agent graph depth |
| `MAX_ITERATIONS` | 10 | Reference for max tool calls |
| Temperature | 0.0 | Deterministic responses |

## Supported Models

| Model | Provider | ToolStrategy | Notes |
|-------|----------|--------------|-------|
| gpt-4o-mini | OpenAI | N/A (uses ProviderStrategy) | Best structured output |
| gpt-oss:latest | Ollama | ✅ | Local, tools capable |
| qwen3.5:27b | Ollama | ✅ | Best for agents |
| qwen3:8b | Ollama | ✅ | Good balance |
| llama3.1:8b | Ollama | ✅ | Reliable |
| gemma3:12b | Ollama | ⚠️ | Limited tool support |

## Usage

```bash
# Install dependencies
uv sync

# Run the application
python main.py
```

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```
