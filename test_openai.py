# test_openai.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env")
    exit(1)

print(f"✓ API Key found: {api_key[:20]}...{api_key[-4:]}")
print(f"✓ Key length: {len(api_key)} characters")

# Test API call
try:
    client = OpenAI(api_key=api_key)

    print("\n🔄 Testing OpenAI API connection…")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello, OpenAI test successful!' and nothing else."}
        ],
        max_tokens=20,
    )

    text = response.choices[0].message["content"]

    print("\n✅ SUCCESS! OpenAI key is working.")
    print(f"📨 Response: {text}")

except Exception as e:
    print(f"\n❌ API ERROR: {e}")
    print("\nPossible issues:")
    print("  1. Invalid API key")
    print("  2. Incorrect model name")
    print("  3. Network issue")
    print("  4. No billing enabled")
    print("\n💡 Check your billing at https://platform.openai.com/account/billing")
