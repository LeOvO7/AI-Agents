import os
import requests
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


@tool
def get_weather_forecast(city: str) -> str:
    """
    Retrieves detailed weather forecast for a specific city.
    
    This tool performs two steps:
    1. Geocoding: Converts city name to latitude/longitude.
    2. One Call API 3.0: Fetches current weather and daily summaries.
    
    Args:
        city: The name of the city (e.g., "Shanghai", "London", "Tokyo").
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not found in environment variables."

    # city -> lat/lon
    geo_url = "http://api.openweathermap.org/geo/1.0/direct"
    try:
        geo_res = requests.get(geo_url, params={"q": city, "limit": 1, "appid": api_key})
        geo_data = geo_res.json()
        
        if not geo_data:
            return f"Error: Could not find city '{city}'. Please check the spelling."
            
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        found_name = geo_data[0]["name"]
        country = geo_data[0]["country"]
        
    except Exception as e:
        return f"Geocoding service connection failed: {str(e)}"

    # openwheather
    onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",  
        "lang": "en",       
        "exclude": "minutely,hourly,alerts" 
    }
    
    try:
        weather_res = requests.get(onecall_url, params=params)
        
        if weather_res.status_code == 401:
            return "API Error: 401 Unauthorized. Please verify your 'One Call by Call' subscription is active on OpenWeatherMap."
            
        weather_res.raise_for_status()
        data = weather_res.json()
        
        # 
        report = []
        report.append(f"【Weather Report for {found_name}, {country}】")
        
        # 1. Current Weather
        current = data.get("current", {})
        temp = current.get("temp", "N/A")
        desc = current.get("weather", [{}])[0].get("description", "Unknown")
        report.append(f"Current Conditions: {desc}, Temperature: {temp}°C")
        
        # 2. Forecast 
        report.append("\nForecast for the next 3 days:")
        daily_list = data.get("daily", [])[:3]
        
        for day in daily_list:
            dt = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
            min_temp = day["temp"]["min"]
            max_temp = day["temp"]["max"]
            summary = day.get("summary", day["weather"][0]["description"])
            report.append(f"- {dt}: {summary} (Temp: {min_temp}°C ~ {max_temp}°C)")
            
        return "\n".join(report)

    except Exception as e:
        return f"Failed to retrieve weather data: {str(e)}"

# Register tools
tools = [get_weather_forecast]

# ----------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize 
model = ChatOpenAI(model="gpt-4o", temperature=0.5)
model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")
app = workflow.compile()

# ---------------------------------

def run_agent():
    print("="*60)
    print("Smart Travel Planner")
    print("="*60)
    
    # Check keys silently
    if not os.environ.get("OPENWEATHER_API_KEY"):
        print("Error: OPENWEATHER_API_KEY environment variable is missing.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is missing.")
        return

    dest = input("Enter Destination : ")
    date = input("Enter Travel Dates : ")
    
    initial_prompt = (
        f"I am traveling to {dest} on {date}. "
        "Please check the weather forecast for that location using the available tool. "
        "Based STRICTLY on the weather conditions and summary provided by the tool, generate a travel guide including:\n"
        "1. Clothing Recommendations (based on temperature and conditions).\n"
        "2. Packing Essentials (e.g., umbrella, sunscreen, specific gear).\n"
        "3. Activity Recommendations (3 specific activities suitable for the weather).\n\n"
        "Please provide the response in English, acting as a helpful local guide."
    )
    
    print(f"\n Agent is thinking and querying the API...\n")
    
    inputs = {"messages": [HumanMessage(content=initial_prompt)]}
    
    try:
        for event in app.stream(inputs, stream_mode="values"):
            message = event["messages"][-1]
            
            if isinstance(message, AIMessage) and message.tool_calls:
                print(f"  Agent is calling tool: {message.tool_calls[0]['name']}")
                
            elif isinstance(message, AIMessage) and message.content:
                print("\n" + "="*60)
                print(" Travel Guide")
                print("="*60)
                print(message.content)
    except Exception as e:
        print(f"\n An error occurred: {e}")

if __name__ == "__main__":
    run_agent()