### The code originates from ipynb notebooks; this breakdown is for review purposes only.

import ipywidgets as widgets
from IPython.display import display, clear_output
import base64
import ollama
import operator
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_image: str

def call_vlm(state: AgentState):
    messages = state['messages']
    current_image = state.get('current_image')
    last_user_msg = messages[-1].content

    history = ""
    if len(messages) > 1:
        history = "History:\n" + "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
             for m in messages[:-1]]
        )

    prompt = f"{history}\n\nQuestion: {last_user_msg}" if history else last_user_msg

    try:
        response = ollama.chat(
            model='llava',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [current_image] if current_image else []
            }]
        )
        ai_response = response['message']['content']
    except Exception as e:
        ai_response = f"Error: {str(e)}"

    return {"messages": [AIMessage(content=ai_response)]}

workflow = StateGraph(AgentState)
workflow.add_node("vlm_node", call_vlm)
workflow.set_entry_point("vlm_node")
workflow.add_edge("vlm_node", END)
app = workflow.compile()

# UI 
chat_state = {"messages": [], "current_image": None}

title = widgets.HTML("<h2>🖼️ LLaVA Vision Chat</h2>")
upload_btn = widgets.FileUpload(accept='image/*', multiple=False, description='1. Upload Image')
image_out = widgets.Image(width=300)
chat_log = widgets.Output(layout={'border': '1px solid #ccc', 'height': '300px', 'overflow_y': 'auto', 'padding': '10px'})
text_input = widgets.Text(placeholder='2. Ask a question about the image...', layout={'width': '70%'})
send_btn = widgets.Button(description='Send', button_style='primary')
status_label = widgets.Label(value="")

def on_upload(change):
    if upload_btn.value:
        uploaded = upload_btn.value[0] if isinstance(upload_btn.value, (list, tuple)) else list(upload_btn.value.values())[0]
        content = uploaded['content']
        
        image_out.value = content
        chat_state['current_image'] = base64.b64encode(content).decode('utf-8')
        
        with chat_log:
            print("--- 📸 Image uploaded successfully ---")

upload_btn.observe(on_upload, names='value')

def on_send(b):
    user_msg = text_input.value.strip()
    if not user_msg: return
    
    text_input.value = ""
    with chat_log:
        print(f"👤 You: {user_msg}")
    
    status_label.value = " LLaVA is thinking..."
    
    input_state = {
        "messages": chat_state["messages"] + [HumanMessage(content=user_msg)],
        "current_image": chat_state["current_image"]
    }
    
    try:
        result = app.invoke(input_state)
        chat_state["messages"] = result["messages"]
        ai_reply = result["messages"][-1].content
        
        with chat_log:
            print(f" LLaVA: {ai_reply}\n")
    except Exception as e:
        with chat_log:
            print(f" Error: {e}\n")
            
    status_label.value = ""

send_btn.on_click(on_send)
text_input.on_submit(on_send)

# Display
ui = widgets.VBox([
    title,
    upload_btn,
    image_out,
    chat_log,
    status_label,
    widgets.HBox([text_input, send_btn])
])

display(ui)