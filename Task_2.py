import os
from openai import OpenAI

try:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say: Working!"}],
    max_tokens=5)
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")