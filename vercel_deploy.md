# Deploy to Vercel (Free Tier)

## Why Vercel?
- ✅ **Unlimited free deployments**
- ✅ **Global CDN** for fast loading
- ✅ **Automatic HTTPS** and custom domains
- ✅ **Zero configuration** for Python apps
- ✅ **GitHub integration** with auto-deployments

## Step 1: Install Vercel CLI (Optional)
```bash
npm i -g vercel
```

## Step 2: Deploy from GitHub
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Click "New Project"
4. Import your `phishing-detector` repository
5. Vercel auto-detects Python and configures everything
6. Click "Deploy"

## Step 3: Set Environment Variables
In Vercel dashboard → Settings → Environment Variables:
- `GEMINI_API_KEY`: Your Gemini API key (optional)

## Step 4: Get Your URL
Vercel gives you a URL like:
- `https://phishing-detector-kyne0328.vercel.app`
- Or your custom domain if configured

## Benefits:
- ✅ **Unlimited free tier**
- ✅ **Global edge network**
- ✅ **Automatic scaling**
- ✅ **Preview deployments** for every PR
- ✅ **Custom domains** support

## Note:
Vercel is optimized for serverless functions, so your app will work great for the phishing detection API!
