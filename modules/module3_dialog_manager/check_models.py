import google.generativeai as genai
import os

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)

print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
