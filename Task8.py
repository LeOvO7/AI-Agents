import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant. Be concise."

USE_HISTORY = True 
MAX_HISTORY = 6 

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16, # Fixed warning: used dtype instead of torch_dtype
    device_map="auto"
)

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

print("\n" + "="*50)
print(f"Chat Started | History: {'ON' if USE_HISTORY else 'OFF'}")
print("="*50 + "\n")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']: break
    if not user_input: continue

    chat_history.append({"role": "user", "content": user_input})

    if USE_HISTORY:
        current_context = [chat_history[0]] + chat_history[1:][-MAX_HISTORY:]
    else:
        current_context = [chat_history[0], chat_history[-1]]

    # Fixed: Explicitly handle dictionary output from apply_chat_template
    model_inputs = tokenizer.apply_chat_template(
        current_context,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    input_ids = model_inputs["input_ids"]

    print("Assistant: ", end="", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # Calculate offset using the extracted input_ids tensor
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(response)
    chat_history.append({"role": "assistant", "content": response})