import math
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ============================================
# PART 1: Define Tools 
# ============================================

@tool
def get_weather(location: str) -> str:
    print(f"  [Tool Executing] Checking weather for {location}...") 
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@tool
def calculate(expression: str) -> str:
    print(f"  [Tool Executing] Calculating {expression}...") 
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, safe_env)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def count_letter(text: str, letter: str) -> str:
    print(f"  [Tool Executing] Counting '{letter}' in text...") 
    if len(letter) != 1: return "Error: Single char only."
    return str(text.lower().count(letter.lower()))

@tool
def reverse_text(text: str) -> str:
    print(f"  [Tool Executing] Reversing text...") 
    return text[::-1]

# List of tools
tools = [get_weather, calculate, count_letter, reverse_text]

# ============================================
# PART 2: Define LangGraph State & Nodes
# ============================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# 1
def agent_node(state: AgentState):
    # Invoke the LLM with the current state history
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 2
tool_node = ToolNode(tools)

# ============================================
# PART 3: Build the Graph
# ============================================

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Add Edges
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition, 
)

workflow.add_edge("tools", "agent")

# Checkpointing
checkpointer = MemorySaver()

# Graph
app = workflow.compile(checkpointer=checkpointer)

# ============================================
# PART 4: Simulation 
# ============================================

def run_chat_session(thread_id, user_input):
    """Runs a single turn of conversation using the persistent graph."""
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\nUser ({thread_id}): {user_input}")
    print("-" * 50)
    
    inputs = {"messages": [HumanMessage(content=user_input)]}

    for event in app.stream(inputs, config=config):
        for key, value in event.items():
            if key == "agent":
                msg = value["messages"][0]
                if msg.tool_calls:
                    print(f"Agent decides to call: {[t['name'] for t in msg.tool_calls]}")
                else:
                    print(f"Agent Output: {msg.content}")
            elif key == "tools":
                print(f"Tool Output generated.")

if __name__ == "__main__":
    # Setting Context
    print("=== SESSION START: Thread ID 'user_123' ===")
    
    # 1. Provide some context
    run_chat_session("user_123", "Hi, my name is Alice and I am in Tokyo.")
    
    # 2. Ask a question that requires context 
    run_chat_session("user_123", "What is the weather like where I am?")
    
    print("\n=== SYSTEM RESTART / CRASH SIMULATION ===")
    print("(Simulating that the script ended and started again...)")
    
    # Recovery
    print("=== SESSION RESUME: Thread ID 'user_123' ===")
    
    # 3. Ask a complex multi-step question referencing previous context
    run_chat_session("user_123", 
        "Since I'm in Tokyo, count how many 'o's are in the city name, "
        "and tell me if that number is equal to 2 + 1.")