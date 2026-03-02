# Topic 1: Running an LLM

---

## Project Structure


```text
.
├── README.md               # Documentation
├── llama_mmlu_eval.py      # Original file
├── simple_chat_agent.py    # Original file
├── Output.txt              # Execution logs
├── Task4_1.py              # Using GPU and no quantization
├── Task4_2.py              # Using GPU and 4-bit quantization
├── Task4_3.py              # Using GPU and 8-bit quantization
├── Task5.py                # Three-model test
└── Task8.py                # Historical record test

```


### Evaluation Summary(Task6)

| Model | Acc | Real (s) | CPU (s) | GPU (s) |
| :--- | :--- | :--- | :--- | :--- |
| **meta-llama/Llama-3.2-1B-Instruct** | 41.53% | 123.26 | 85.36 | 123.26 |
| **Qwen/Qwen2.5-1.5B-Instruct** | 52.96% | 180.46 | 175.08 | 180.46 |
| **TinyLlama/TinyLlama-1.1B-Chat-v1.0** | 23.85% | 144.11 | 133.57 | 144.11 |
