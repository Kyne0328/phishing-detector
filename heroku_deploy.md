# Deploy to Heroku (Free Tier)

## Step 1: Install Heroku CLI
Download from [devcenter.heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

## Step 2: Prepare Files
Create `Procfile`:
```
web: python app.py
```

Create `runtime.txt`:
```
python-3.12.0
```

## Step 3: Deploy
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-phishing-detector

# Set environment variables
heroku config:set GEMINI_API_KEY=your_key_here

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## Benefits:
- ✅ Free tier available
- ✅ Easy CLI deployment
- ✅ Add-ons marketplace
- ✅ Built-in monitoring
