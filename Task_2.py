import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

def get_device():
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

class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    llm_response: str
    # 1
    verbose: bool

def create_llm():
    device = get_device()

    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  
        do_sample=True,      
        temperature=0.7,     
        top_p=0.95,          
        pad_token_id=tokenizer.eos_token_id,  
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    print("Model loaded successfully!")
    return llm

def create_graph(llm):
    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        # 1
        if state.get("verbose", False):
            print(f"[Trace] Entering node: get_user_input. Current state: {state}")

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

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,       
                # 1
                "verbose": verbose
            }

        return {
            "user_input": user_input,
            "should_exit": False,          
            # 1
            "verbose": verbose
        }

    # =========================================================================
    # NODE 2: call_llm
    # =========================================================================
    def call_llm(state: AgentState) -> dict:
        # 1
        if state.get("verbose", False):
            print(f"[Trace] Entering node: call_llm. Input: {state['user_input']}")

        user_input = state["user_input"]

        prompt = f"User: {user_input}\nAssistant:"

        print("\nProcessing your input...")

        response = llm.invoke(prompt)

        return {"llm_response": response}

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        # 1
        if state.get("verbose", False):
            print(f"[Trace] Entering node: print_response. Response length: {len(state['llm_response'])}")

        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        
        # 2
        if not state["user_input"]:
            return "get_user_input"

        return "call_llm"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)
    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",      
        route_after_input,      
        {
            "call_llm": "call_llm",  
            # 2
            "get_user_input": "get_user_input",
            END: END                  
        }
    )

    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")
    graph = graph_builder.compile()

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def main():
    print("=" * 50)
    print("LangGraph Simple Agent with Llama-3.2-1B-Instruct")
    print("=" * 50)
    print()

    llm = create_llm()

    print("\nCreating LangGraph...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        # 1
        "verbose": False
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()