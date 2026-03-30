# setup_api.py
import os
import getpass

print("🔐 Setting up API Key for Onna's SmartQA")
print("-" * 40)

# Get new API key
api_key = getpass.getpass("Paste your new Gemini API Key: ")

# Save to .env file
with open(".env", "w") as f:
    f.write(f"GEMINI_API_KEY={api_key}\n")

print("✅ API Key saved to .env file")
print("⚠️  IMPORTANT: Do not share this file or commit it to GitHub!")