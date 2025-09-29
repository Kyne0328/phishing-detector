# Deploy to Google Cloud Run (Free Tier)

## Step 1: Create Dockerfile
Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
```

## Step 2: Deploy
```bash
# Install Google Cloud CLI
# Then run:
gcloud run deploy phishing-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here
```

## Benefits:
- ✅ Free tier: 2 million requests/month
- ✅ Pay only for what you use
- ✅ Auto-scaling
- ✅ Global CDN
