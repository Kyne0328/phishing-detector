# Gemini AI Integration Setup

This app now includes AI-powered interpretation of URL analysis results using Google's Gemini AI. Here's how to set it up:

## 1. Get a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

## 2. Set the API Key

### Option A: Environment Variable (Recommended)
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"
```

### Option B: Create a .env file
Create a `.env` file in the project directory:
```
GEMINI_API_KEY=your_api_key_here
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the App

```bash
python app.py
```

## Features

With Gemini AI enabled, you'll get:

- **🤖 Intelligent Analysis**: AI explains what the results mean in plain English
- **📊 Confidence Explanation**: Clear explanation of what the confidence level means
- **🔍 Key Indicators**: AI identifies the most important features that influenced the decision
- **💡 Recommendations**: Actionable advice based on the analysis

## Fallback Mode

If no API key is provided, the app will still work with a built-in fallback interpretation system that provides basic explanations without requiring Gemini AI.

## Troubleshooting

- **"GEMINI_API_KEY not found"**: Make sure you've set the environment variable correctly
- **"Google Generative AI not available"**: Run `pip install google-generativeai`
- **API errors**: Check that your API key is valid and has sufficient quota

The app will automatically detect if Gemini AI is available and use it when possible, falling back to the built-in interpreter when not available.
