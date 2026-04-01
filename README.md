# Agent flow

ReAct-style loop: the model **thinks**, may **act** on a **tool**, receives an **observation**, and repeats until it can answer.

```mermaid
flowchart TB
    subgraph react[" "]
        Q([Query])
        subgraph decide[" "]
            direction LR
            T{Thought}
            A([Answer])
        end
        TO([Tool])

        Q --> T
        T -.-> A
        T -->|Action| TO
        TO -->|Observation| T
    end

    classDef queryAnswer fill:#4a90d9,stroke:#2c5282,color:#fff
    classDef thought fill:#f5d742,stroke:#b7791f,color:#000
    classDef tool fill:#a8d8f0,stroke:#2c5282,color:#000

    class Q,A queryAnswer
    class T thought
    class TO tool
```

The dashed arrow from **Thought** to **Answer** is the path when the agent finishes without another tool call.
