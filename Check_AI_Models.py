# test_models.py
import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="AIzaSyCQqjEc1_nM_0AkOuBKFeOU33MfetV9IM8")

# List available models
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"  Supported methods: {model.supported_generation_methods}")
    print()