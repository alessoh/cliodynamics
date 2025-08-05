#!/usr/bin/env python3
"""
Test script to verify API keys are working correctly
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI API key"""
    print("\n=== Testing OpenAI API ===")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        return False
    
    print(f"✓ API key found (length: {len(api_key)})")
    print(f"✓ Key format: {api_key[:7]}...{api_key[-4:]}")
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )
        print(f"✓ API call successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return False

def test_anthropic():
    """Test Anthropic API key"""
    print("\n=== Testing Anthropic/Claude API ===")
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env file")
        return False
    
    # Remove any whitespace
    api_key = api_key.strip()
    
    print(f"✓ API key found (length: {len(api_key)})")
    print(f"✓ Key format: {api_key[:10]}...{api_key[-4:]}")
    
    # Check key format
    if not api_key.startswith('sk-ant-api'):
        print("⚠️  Warning: Key doesn't start with expected 'sk-ant-api' prefix")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Using cheaper model for testing
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'API test successful'"}]
        )
        print(f"✓ API call successful: {response.content[0].text}")
        return True
    except anthropic.AuthenticationError as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("\nPossible issues:")
        print("1. Check that your API key is correct")
        print("2. Make sure there are no extra spaces or quotes around the key")
        print("3. Verify the key is active in your Anthropic console")
        print("4. The key should look like: sk-ant-api03-xxxxx...")
        return False
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return False

def check_env_file():
    """Check .env file exists and format"""
    print("\n=== Checking .env file ===")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("\nCreate a .env file with:")
        print("OPENAI_API_KEY=your-openai-key")
        print("ANTHROPIC_API_KEY=your-anthropic-key")
        return False
    
    print("✓ .env file found")
    
    # Check for common issues
    with open('.env', 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Check for quotes (which shouldn't be there)
                if value.startswith('"') or value.startswith("'"):
                    print(f"⚠️  Warning: {key} has quotes around it. Remove the quotes!")
                
                # Check for spaces
                if ' ' in value and key.endswith('_KEY'):
                    print(f"⚠️  Warning: {key} contains spaces. This might be an error!")
    
    return True

if __name__ == "__main__":
    print("API Key Testing Utility")
    print("=" * 50)
    
    # Check environment file
    env_ok = check_env_file()
    
    # Test APIs
    openai_ok = test_openai()
    anthropic_ok = test_anthropic()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Environment file: {'✓' if env_ok else '❌'}")
    print(f"OpenAI API: {'✓' if openai_ok else '❌'}")
    print(f"Anthropic API: {'✓' if anthropic_ok else '❌'}")
    
    if not anthropic_ok:
        print("\n💡 You can still use the system with just OpenAI!")
        print("   Run: python main.py --start-year 1900 --end-year 2050")