import ollama
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "llama3.2:1b"
SUBJECT = "elementary_mathematics" 
LIMIT = 5  

def format_prompt(question, choices):
    options = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(options, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer with only the single letter (A, B, C, or D) of the correct answer."
    return prompt

def main():
    print(f"--- Program 1 Starting: {SUBJECT} ---")
    
    dataset = load_dataset("cais/mmlu", SUBJECT, split=f"test[:{LIMIT}]")
    
    processed = 0
    for example in tqdm(dataset, desc="Math Progress"):
        try:
            prompt = format_prompt(example["question"], example["choices"])
            
            response = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            _ = response['message']['content']
            processed += 1
            
        except Exception as e:
            print(f"Error: {e}")

    print(f"--- Program 1 Finished ({processed} questions) ---")

if __name__ == "__main__":
    main()