# Phishing Link Detector

A web application that uses hierarchical clustering machine learning to detect phishing URLs. Built with Flask and featuring a sleek dark mode UI.

## Features

- **Hierarchical Clustering Model**: Uses scikit-learn's AgglomerativeClustering for URL analysis
- **Dark Mode UI**: Modern, responsive design with gradient accents
- **Real-time Analysis**: Instant URL checking with confidence scores
- **Animated Results**: Smooth fade-in animations for better UX
- **Feature Extraction**: Analyzes URL characteristics like length, special characters, suspicious patterns, and more

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter a URL in the input field and click "Check" to analyze it

## How It Works

The application uses hierarchical clustering to classify URLs as either safe or phishing attempts. It uses an optimized set of 15 most important features from the original 30-feature dataset:

**Top 5 Most Important Features:**
1. **SSLfinal_State** (31.97%) - SSL certificate state
2. **URL_of_Anchor** (25.00%) - URL of anchor analysis  
3. **web_traffic** (7.08%) - Web traffic analysis
4. **having_Sub_Domain** (6.95%) - Number of subdomains
5. **Links_in_tags** (4.39%) - Links in tags analysis

The model is trained on a real dataset of 11,055 samples with known safe and phishing URLs, achieving 97% accuracy with just 15 optimized features instead of all 30.

## UI/UX Features

- **Dark Theme**: Gray-900 background with light text
- **Gradient Headlines**: Blue/teal gradient for main headings
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Animated spinner during analysis
- **Result Cards**: Color-coded results with confidence scores
- **Smooth Animations**: Fade-in effects and hover animations

## File Structure

```
Hierarchical Clustering/
├── app.py                 # Main Flask application (optimized with 15 features)
├── arff_parser.py         # ARFF dataset parser
├── templates/
│   └── index.html        # HTML template with embedded CSS/JS
├── test_model.py         # Model testing script
├── Training Dataset.arff # Training dataset
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies

- Flask 2.3.3
- NumPy 1.24.3
- scikit-learn 1.3.0
- requests 2.31.0
- Werkzeug 2.3.7

## Note

This is a demonstration application. For production use, you would need:
- A larger, more diverse training dataset
- Regular model retraining
- Additional security features
- Proper error handling and logging
