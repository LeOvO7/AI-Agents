# Topic 3: Agent Tool Use

---

## Project Structure

```text
.
├── Task_1_program1.py
├── Task_1_program2.py
├── Task_2.py
├── Task_3.py
├── Task_4.py
├── Task_5.py
├── langgraph-tool-handling.py  # Original file
├── llama_mmlu_eval.py          # Original file
├── manual-tool-handling.py     # Original file
├── output.txt                  # Output
└── README.md                   # Documentation
```
## Task_5
### System Architecture

The following Mermaid diagram illustrates the **cyclic graph** architecture implemented using LangGraph. The system consists of an Agent node (LLM) and a Tools node, connected by conditional edges. State is persisted to a `MemorySaver` checkpointer after every step, allowing for session recovery.

```mermaid
graph TD
    %% Nodes
    __start__([Start]) --> agent
    agent["Agent Node<br>(GPT-4o-mini)"]
    tools["Tool Node<br>(Python Functions)"]
    __end__([End])

    %% Edges
    agent -- "Calls Tool" --> tools
    tools -- "Returns Output" --> agent
    agent -- "Final Answer" --> __end__

    %% Persistence Layer Visualization
    subgraph Persistence [Checkpointing Layer]
        direction TB
        db[("MemorySaver<br>In-Memory DB")]
        agent -.-> db
        tools -.-> db
    end
    
    %% Styling
    style agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style tools fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style db fill:#e0e0e0,stroke:#616161,stroke-dasharray: 5 5
```
## Task_6
The opportunity lies in parallelizing independent tool calls. In the example, the agent executed the tasks sequentially: it counted the letters first, waited for the result, and then calculated "2 + 1". Since the calculation of "2 + 1" does not depend on the letter count, the Agent should have requested both the count_letter and calculate tools in the very first turn. This would have compressed two separate round-trips into a single execution step, significantly improve efficiency.

