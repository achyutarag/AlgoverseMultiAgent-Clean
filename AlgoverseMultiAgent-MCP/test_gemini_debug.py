#!/usr/bin/env python3
"""
Debug script to test Google Gemini API and list available models.
This will help us identify the correct model name to use.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_api():
    """Test Gemini API and list available models."""
    print("🔍 Testing Google Gemini API...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key or credentials
    api_key = os.getenv("GOOGLE_API_KEY")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if api_key:
        print(f"🔑 Using API Key: {api_key[:10]}...{api_key[-4:]}")
        genai.configure(api_key=api_key)
        print("✅ Configured with API Key")
    elif creds_path:
        print(f"📁 Credentials path: {creds_path}")
        if not os.path.exists(creds_path):
            print(f"❌ Credentials file not found: {creds_path}")
            return
        print(f"✅ Credentials file exists: {creds_path}")
        genai.configure()
        print("✅ Configured with Service Account")
    else:
        print("❌ Neither GOOGLE_API_KEY nor GOOGLE_APPLICATION_CREDENTIALS set!")
        print("Please set one in your .env file")
        return
    
    try:
        # Configure Gemini
        print("\n🔧 Testing Gemini API...")
        print("✅ Gemini API configured successfully")
        
        # List available models
        print("\n📋 Listing available models...")
        models = list(genai.list_models())
        
        print(f"Found {len(models)} models:")
        print("-" * 30)
        
        for model in models:
            print(f"📌 {model.name}")
            if hasattr(model, 'display_name'):
                print(f"   Display Name: {model.display_name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"   Methods: {model.supported_generation_methods}")
            print()
        
        # Test with Gemini Flash Lite (latest)
        print("🧪 Testing with models/gemini-flash-lite-latest...")
        
        test_model = "models/gemini-flash-lite-latest"
        
        try:
            print(f"🎯 Testing with model: {test_model}")
            model = genai.GenerativeModel(test_model)
            
            response = model.generate_content("Hello! Can you say 'API test successful'?")
            print(f"✅ Response: {response.text}")
        except Exception as model_error:
            print(f"❌ Error testing model: {model_error}")
            print("Trying to find any available model...")
            
            # Fallback: Try to find any model that supports generateContent
            available_models = list(genai.list_models())
            for model in available_models:
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        test_model = model.name
                        print(f"🎯 Found alternative model: {test_model}")
                        model = genai.GenerativeModel(test_model)
                        response = model.generate_content("Hello! Can you say 'API test successful'?")
                        print(f"✅ Response: {response.text}")
                        break
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_gemini_api()



