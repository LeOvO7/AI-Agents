# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a very simple agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It passes the line to the LLM llama-3.2-1B-Instruct, then prints the
# what the LLM returns as text to stdout.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
# The program gets the LLM llama-3.2-1B-Instruct from Hugging Face and wraps
# it for LangChain using HuggingFacePipeline.
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
# 5
from typing import TypedDict, Optional, Annotated, List
# 5
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
# 5
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: List of messages maintaining chat history (user, ai, system)
    - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - verbose: Boolean flag for trace logging
    """
    # 5
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    should_exit: bool
    # 1
    verbose: bool
    # 6
    # Track the last active model for routing purposes if needed, though input based routing is used here
    last_model: Optional[str]

# 3
def create_specific_llm(model_id):
    """
    Create and configure a specific LLM using HuggingFace's transformers library.
    """
    # Get the optimal device for inference
    device = get_device()

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model itself with appropriate settings for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Maximum tokens to generate in response
        do_sample=True,      # Enable sampling for varied responses
        temperature=0.7,     # Controls randomness (lower = more deterministic)
        top_p=0.95,          # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
        return_full_text=False
    )

    # Wrap the HuggingFace pipeline for use with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    print(f"Model {model_id} loaded successfully!")
    return llm

# 6
def format_history_for_model(messages: List[BaseMessage], current_model_name: str) -> List[dict]:
    """
    Transforms the chat history into a format suitable for the specific model.
    
    Logic:
    1. System Prompt: Adds a custom system prompt defining the participants.
    2. Role Mapping:
       - 'Human: ...' -> role: 'user'
       - 'CurrentModel: ...' -> role: 'assistant'
       - 'OtherModel: ...' -> role: 'user' (The other model is treated as a user/interlocutor)
    """
    formatted_messages = []
    
    # Define participants for the system prompt
    participants = "Human, Llama, and Qwen"
    system_prompt = f"You are {current_model_name}. You are participating in a conversation between {participants}."
    formatted_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        content = msg.content
        role = "user" # Default to user

        # Determine the role based on the content prefix and the current model
        if content.startswith("Human:"):
            role = "user"
        elif content.startswith(f"{current_model_name}:"):
            # If the message was generated by the current model, it is 'assistant'
            role = "assistant"
        else:
            # If generated by the other model (e.g., Llama seeing Qwen's msg), treat as 'user'
            role = "user"
        
        formatted_messages.append({"role": role, "content": content})
    
    return formatted_messages

# 5
def create_graph(llama_llm, qwen_llm, checkpointer):
    """
    Create the LangGraph state graph.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin.
        """
        # 1
        if state.get("verbose", False):
            print(f"[Trace] Entering node: get_user_input. Current state keys: {state.keys()}")

        # Display banner before each prompt
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        # 2
        user_input = input().strip()

        # 1
        verbose = state.get("verbose", False)
        # 2
        if user_input == "verbose":
            verbose = True
            print("Verbose mode enabled.")
        # 2
        elif user_input == "quiet":
            verbose = False
            print("Quiet mode enabled.")

        # Check if user wants to exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                # 1
                "verbose": verbose
            }

        # Any input (including empty) - continue to LLM
        # 5
        # If valid input, add to message history
        # 6
        # Prepend "Human: " to explicitly identify the speaker in the unified history
        messages_update = []
        if user_input and user_input not in ["verbose", "quiet"]:
             messages_update = [HumanMessage(content=f"Human: {user_input}")]

        return {
            "user_input": user_input,
            "should_exit": False,
            "messages": messages_update,
            # 1
            "verbose": verbose
        }

    # =========================================================================
    # NODE: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        # 1
        if state.get("verbose", False):
            # 5
            print(f"[Trace] Entering node: call_llama. History length: {len(state['messages'])}")

        # 6
        # Generate the formatted prompt where Qwen's messages appear as 'user'
        formatted_messages = format_history_for_model(state['messages'], "Llama")
        
        # 5
        # Use the tokenizer embedded in the pipeline to apply the template
        prompt = llama_llm.pipeline.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3
        print("\nLlama is processing...")
        response = llama_llm.invoke(prompt)

        # 5
        # Return the response as an AIMessage to append to history
        # 6
        # Prepend "Llama: " so future nodes know who said this
        return {"messages": [AIMessage(content=f"Llama: {response.strip()}")]}

    # 6
    # =========================================================================
    # NODE: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[Trace] Entering node: call_qwen. History length: {len(state['messages'])}")

        # 6
        # Generate the formatted prompt where Llama's messages appear as 'user'
        formatted_messages = format_history_for_model(state['messages'], "Qwen")
        
        # Use the tokenizer embedded in the pipeline to apply the template
        prompt = qwen_llm.pipeline.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print("\nQwen is processing...")
        response = qwen_llm.invoke(prompt)

        # Prepend "Qwen: " so future nodes know who said this
        return {"messages": [AIMessage(content=f"Qwen: {response.strip()}")]}

    # =========================================================================
    # NODE: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the responses from the model.
        """
        # 1
        if state.get("verbose", False):
            print(f"[Trace] Entering node: print_response.")

        # 5
        # Get the last message, which should be the AI response
        last_message = state['messages'][-1]
        
        # 6
        # Cleanly display the response (strip the "Name: " prefix for display if desired, 
        # but here we keep it to be clear who spoke)
        if isinstance(last_message, AIMessage):
            print("\n" + "-" * 50)
            print("Response:")
            print("-" * 50)
            print(last_message.content)

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        # Check if user wants to exit
        if state.get("should_exit", False):
            return END
        
        # 2
        # Check if input is empty or just a command
        if not state["user_input"] or state["user_input"] in ["verbose", "quiet"]:
            return "get_user_input"
        
        # 4
        # Check specific trigger for Qwen
        if state["user_input"].lower().startswith("hey qwen"):
            return "call_qwen"
        
        # Default to Llama
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    # 6
    # Re-add Qwen node
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    # Define edges
    graph_builder.add_edge(START, "get_user_input")

    # Conditional edge from input
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            # 6
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    # Converge: Both send results to print_response
    graph_builder.add_edge("call_llama", "print_response")
    # 6
    graph_builder.add_edge("call_qwen", "print_response")

    # Loop back
    graph_builder.add_edge("print_response", "get_user_input")

    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")

def main():
    print("=" * 50)
    # 6
    print("LangGraph Multi-Agent Chat (Llama 3.2 & Qwen 2.5 Shared History)")
    print("=" * 50)
    print()

    # Step 1: Create and configure the LLMs
    # 3
    print("Initializing Llama...")
    llama_llm = create_specific_llm("meta-llama/Llama-3.2-1B-Instruct")
    
    # 6
    # Re-enable Qwen
    print("\nInitializing Qwen...")
    qwen_llm = create_specific_llm("Qwen/Qwen2.5-1.5B-Instruct")

    # Step 2: Build the LangGraph with the LLMs
    print("\nCreating LangGraph...")
    
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        # 6
        graph = create_graph(llama_llm, qwen_llm, checkpointer)
        print("Graph created successfully!")

        print("\nSaving graph visualization...")
        save_graph_image(graph)

        # Step 4: Run the graph
        initial_state: AgentState = {
            # 5
            "messages": [], 
            "user_input": "",
            "should_exit": False,
            # 1
            "verbose": False,
            # 6
            "last_model": None
        }

        config = {"configurable": {"thread_id": "1"}}

        print("\nResuming session (if exists)...")
        graph.invoke(initial_state, config=config)

if __name__ == "__main__":
    main()