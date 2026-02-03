import math
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F",
        "Paris": "Windy, 50°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@tool
def calculate(expression: str) -> str:
    try:
        # Safe environment with math functions
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, safe_env)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def count_letter(text: str, letter: str) -> str:
    """Counts the number of occurrences of a specific letter in a piece of text.
    Case-insensitive.
    Args:
        text: The text to search within.
        letter: The single letter to count.
    """
    if len(letter) != 1:
        return "Error: Please provide a single character to count."
    count = text.lower().count(letter.lower())
    return str(count)

@tool
def reverse_text(text: str) -> str:
    """Reverses the given text string. Useful for word games or encryption.
    Args:
        text: The string to reverse.
    """
    return text[::-1]

# ============================================
# PART 2: Configuration & Tool Mapping
# ============================================

# List of tools
tools = [get_weather, calculate, count_letter, reverse_text]
tool_map = {tool.name: tool for tool in tools}

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    print(f"\nUser Query: {user_query}")
    print("-" * 50)
    
    messages = [
        SystemMessage(content="You are a helpful assistant. You have tools for weather, calculation, counting letters, and reversing text. Use them liberally to solve complex problems."),
        HumanMessage(content=user_query)
    ]
    
    # Agent loop
    for iteration in range(5):
        print(f"--- Turn {iteration + 1} ---")
        
        # Call LLM
        response = llm_with_tools.invoke(messages)
        
        # Tool calls
        if response.tool_calls:
            print(f"LLM Strategy: Calling {len(response.tool_calls)} tool(s)...")
            messages.append(response)
            
            # Execute
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                
                print(f"  ➜ Executing: {function_name} with {function_args}")
                
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"    Result: {result}")
                
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
            
            print("  (Feeding results back to LLM...)")
            
        else:
            # Answer
            print(f"\nAssistant: {response.content}")
            return
            
    print("Warning: Max iterations reached without final answer.")

# ============================================
# PART 4: Complex Test Scenarios
# ============================================

if __name__ == "__main__":
    # Parallel Tool Use (Same Turn)
    print("\n" + "="*60)
    print("TEST 1: Parallel Execution (Counting Letters)")
    print("="*60)
    run_agent("Are there more 's's than 'i's in the phrase 'Mississippi riverboats'?")

    # Sequential Chaining (Multi-step Math)
    print("\n" + "="*60)
    print("TEST 2: Sequential Chaining (Math + Counting)")
    print("="*60)
    run_agent("Calculate the sine of the difference between the number of 'i's and the number of 's's in 'Mississippi riverboats'.")

    # All Tools
    print("\n" + "="*60)
    print("TEST 3: Complex Multi-Step (All Tools)")
    print("="*60)
    query = (
        "1. Get the weather description (e.g. Sunny, Cloudy) for Tokyo. "
        "2. Reverse that word. "
        "3. Count how many times the letter 'r' appears in that reversed word. "
        "4. Calculate that number to the power of 3."
    )
    run_agent(query)