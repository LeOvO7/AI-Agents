import json
import math
from openai import OpenAI

# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression including geometric functions.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, pi
    """
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, safe_env)
        
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ============================================
# PART 2: Describe Tools to the LLM
# ============================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "A calculator that can perform arithmetic and geometric calculations. Use Python syntax. Available functions: sqrt, sin, cos, tan, pi, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'pi * 5**2')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    """
    client = OpenAI()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed. For math problems, write Python expressions."},
        {"role": "user", "content": user_query}
    ]
    
    print(f"User: {user_query}\n")
    
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                result = ""
                if function_name == "get_weather":
                    result = get_weather(**function_args)
                elif function_name == "calculate":
                    result = calculate(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"  Result: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            
            print()
            
        else:
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content
    
    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test 1: Original Weather Tool
    print("="*60)
    print("TEST 1: Weather Tool")
    print("="*60)
    run_agent("What's the weather like in Tokyo?")
    
    # Test 2: New Calculator Tool (Basic)
    print("\n" + "="*60)
    print("TEST 2: Calculator (Basic Math)")
    print("="*60)
    run_agent("What is 153 times 98?")
    
    # Test 3: New Calculator Tool (Geometric)
    print("\n" + "="*60)
    print("TEST 3: Calculator (Geometry)")
    print("="*60)
    run_agent("Calculate the area of a circle with a radius of 10. (Hint: Use pi * r squared)")

    # Test 4: Mixed
    print("\n" + "="*60)
    print("TEST 4: Mixed Intent")
    print("="*60)
    run_agent("If the temperature in San Francisco is X, what is the square root of X?")