#!/usr/bin/env python3
"""
Quick test script to verify a new API key works.
Run this after updating your .env file with a new API key.
"""
from openai import OpenAI
import config

print("="*60)
print("Testing New API Key")
print("="*60)
print(f"Key loaded: {config.OPENAI_API_KEY[:30]}...")
print(f"Key type: {'Project-based' if config.OPENAI_API_KEY.startswith('sk-proj-') else 'User-based'}")
print()

try:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    print("Testing API connection...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello' if you can read this"}],
        max_tokens=10
    )
    
    print("="*60)
    print("✓ SUCCESS! API key is working!")
    print("="*60)
    print(f"Response: {response.choices[0].message.content}")
    print()
    print("You can now run your main.py script!")
    
except Exception as e:
    error_msg = str(e)
    print("="*60)
    print("❌ ERROR: API key test failed")
    print("="*60)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {error_msg}")
    print()
    
    if "invalid_organization" in error_msg.lower() or "organization" in error_msg.lower():
        print("⚠️  This means the API key is tied to an organization.")
        print()
        print("SOLUTION:")
        print("1. Go to: https://platform.openai.com/projects")
        print("2. Make sure you're in YOUR PERSONAL project (not an org project)")
        print("3. Create a new API key in that personal project")
        print("4. Update your .env file with the new key")
    elif "401" in error_msg or "authentication" in error_msg.lower():
        print("⚠️  Authentication failed.")
        print()
        print("SOLUTION:")
        print("1. Double-check the API key in your .env file")
        print("2. Make sure there are no quotes or extra spaces")
        print("3. Make sure the key is complete (very long)")
        print("4. Try creating a fresh API key")
    else:
        print("⚠️  Unexpected error. Check the error message above.")

