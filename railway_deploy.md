# Deploy to Railway (Free Tier)

## Step 1: Prepare for Railway
1. Create a `Procfile` in your project root:
```
web: python app.py
```

2. Update `app.py` to use Railway's port:
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

## Step 2: Deploy
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect it's a Python app
6. Deploy!

## Step 3: Set Environment Variables
In Railway dashboard:
- `GEMINI_API_KEY`: Your Gemini API key (optional)

## Benefits:
- ✅ Free tier: 500 hours/month
- ✅ Automatic deployments from GitHub
- ✅ Custom domain support
- ✅ Built-in monitoring
- ✅ No credit card required
