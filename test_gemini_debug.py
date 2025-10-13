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
    
    # Check if credentials are set
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"📁 Credentials path: {creds_path}")
    
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set!")
        return
    
    if not os.path.exists(creds_path):
        print(f"❌ Credentials file not found: {creds_path}")
        return
    
    print(f"✅ Credentials file exists: {creds_path}")
    
    try:
        # Configure Gemini
        print("\n🔧 Configuring Gemini API...")
        genai.configure()
        print("✅ Gemini API configured successfully")
        
        # List available models
        print("\n📋 Listing available models...")
        models = genai.list_models()
        
        print(f"Found {len(list(models))} models:")
        print("-" * 30)
        
        for model in models:
            print(f"📌 {model.name}")
            if hasattr(model, 'display_name'):
                print(f"   Display Name: {model.display_name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"   Methods: {model.supported_generation_methods}")
            print()
        
        # Test with a simple model
        print("🧪 Testing with a simple model...")
        
        # Try to find a model that supports generateContent
        available_models = list(genai.list_models())
        test_model = None
        
        for model in available_models:
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    test_model = model.name
                    break
        
        if test_model:
            print(f"🎯 Testing with model: {test_model}")
            model = genai.GenerativeModel(test_model)
            
            response = model.generate_content("Hello! Can you say 'API test successful'?")
            print(f"✅ Response: {response.text}")
        else:
            print("❌ No models found that support generateContent")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_gemini_api()



