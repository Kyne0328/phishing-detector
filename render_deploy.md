# Deploy to Render (Free Tier)

## Step 1: Create render.yaml
Create `render.yaml` in your project root:
```yaml
services:
  - type: web
    name: phishing-detector
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: GEMINI_API_KEY
        sync: false
```

## Step 2: Deploy
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect the configuration
6. Deploy!

## Benefits:
- ✅ Free tier: 750 hours/month
- ✅ Automatic SSL certificates
- ✅ Custom domains
- ✅ Zero-downtime deployments
