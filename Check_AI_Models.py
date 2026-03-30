# test_models.py
import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="Give_API_KEY_HERE")  # Replace with your actual API key

# List available models
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"  Supported methods: {model.supported_generation_methods}")
    print()