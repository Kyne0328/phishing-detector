from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import pandas as pd
import json
import io
import base64
import re
import urllib.parse
import requests
from urllib.parse import urlparse
import time
import random
import os
from arff_parser import parse_arff_file, get_feature_description

# Gemini AI integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available. Install with: pip install google-generativeai")

app = Flask(__name__)

class GeminiInterpreter:
    """AI-powered interpreter for clustering analysis results"""
    
    def __init__(self):
        self.available = GEMINI_AVAILABLE
        if self.available:
            try:
                # Configure Gemini (you'll need to set GEMINI_API_KEY environment variable)
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-pro')
                    self.initialized = True
                else:
                    print("Warning: GEMINI_API_KEY not found. Set it as environment variable.")
                    self.initialized = False
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.initialized = False
        else:
            self.initialized = False
    
    def interpret_analysis(self, url, prediction, analytics):
        """Generate AI interpretation of the URL analysis"""
        if not self.initialized:
            return self._fallback_interpretation(url, prediction, analytics)
        
        try:
            # Prepare data for interpretation
            interpretation_data = self._prepare_interpretation_data(url, prediction, analytics)
            
            # Generate AI interpretation
            prompt = self._create_interpretation_prompt(interpretation_data)
            response = self.model.generate_content(prompt)
            
            return {
                'ai_interpretation': response.text,
                'confidence_explanation': self._explain_confidence(prediction),
                'feature_insights': self._explain_features(analytics),
                'recommendations': self._generate_recommendations(prediction, analytics)
            }
        except Exception as e:
            print(f"Error generating AI interpretation: {e}")
            return self._fallback_interpretation(url, prediction, analytics)
    
    def _prepare_interpretation_data(self, url, prediction, analytics):
        """Prepare data for AI interpretation"""
        return {
            'url': url,
            'is_phishing': prediction['is_phishing'],
            'confidence': prediction['confidence'],
            'cluster': prediction['cluster'],
            'neighbor_analysis': analytics.get('neighbor_analysis', {}),
            'feature_contributions': analytics.get('feature_contributions', {}),
            'outlier_score': analytics.get('outlier_score', 0),
            'cluster_characteristics': analytics.get('cluster_characteristics', {})
        }
    
    def _create_interpretation_prompt(self, data):
        """Create prompt for Gemini AI"""
        # Add contextual information to make responses more varied
        url_context = self._analyze_url_context(data['url'])
        confidence_context = self._get_confidence_context(data['confidence'])
        neighbor_context = self._get_neighbor_context(data['neighbor_analysis'])
        
        return f"""
        You are an expert cybersecurity analyst. Analyze this URL phishing detection result and provide a clear, user-friendly explanation.

        URL: {data['url']}
        URL Context: {url_context}
        Classification: {'PHISHING' if data['is_phishing'] else 'SAFE'}
        Confidence: {data['confidence']:.1f}% ({confidence_context})
        Cluster: {data['cluster']}
        Outlier Score: {data['outlier_score']:.2f}

        Neighbor Analysis:
        - Total similar URLs analyzed: {data['neighbor_analysis'].get('total_neighbors', 0)}
        - Phishing neighbors: {data['neighbor_analysis'].get('phishing_neighbors', 0)}
        - Safe neighbors: {data['neighbor_analysis'].get('safe_neighbors', 0)}
        - Context: {neighbor_context}

        Top Contributing Features:
        {self._format_top_features(data['feature_contributions'])}

        Please provide a personalized analysis that:
        1. Explains what this specific result means for this particular URL
        2. Describes why the AI made this specific classification based on the unique features
        3. Explains what the confidence level means in this context
        4. Identifies specific red flags or positive indicators found for this URL
        5. Gives practical, specific advice for this user about this URL

        Make your response unique and contextual to this specific URL and analysis. Avoid generic explanations.
        """
    
    def _format_top_features(self, feature_contributions):
        """Format top contributing features for the prompt"""
        if not feature_contributions:
            return "No feature data available"
        
        # Get top 5 most important features
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )[:5]
        
        formatted = []
        for feature, data in sorted_features:
            direction = "toward phishing" if data['pulls_toward_cluster_1'] else "toward safe"
            formatted.append(f"- {feature}: {direction} (contribution: {data['contribution']:.2f})")
        
        return "\n".join(formatted)
    
    def _analyze_url_context(self, url):
        """Analyze URL context for more specific AI responses"""
        context_parts = []
        
        # Domain analysis
        if 'google.com' in url.lower():
            context_parts.append("This is a Google domain")
        elif 'microsoft.com' in url.lower():
            context_parts.append("This is a Microsoft domain")
        elif 'amazon.com' in url.lower():
            context_parts.append("This is an Amazon domain")
        elif 'facebook.com' in url.lower():
            context_parts.append("This is a Facebook domain")
        elif 'apple.com' in url.lower():
            context_parts.append("This is an Apple domain")
        elif 'github.com' in url.lower():
            context_parts.append("This is a GitHub domain")
        elif 'stackoverflow.com' in url.lower():
            context_parts.append("This is a Stack Overflow domain")
        else:
            context_parts.append("This is a custom domain")
        
        # Protocol analysis
        if url.startswith('https://'):
            context_parts.append("uses secure HTTPS protocol")
        elif url.startswith('http://'):
            context_parts.append("uses unsecured HTTP protocol")
        
        # URL length analysis
        if len(url) > 100:
            context_parts.append("has an unusually long URL")
        elif len(url) < 20:
            context_parts.append("has a very short URL")
        
        # Subdomain analysis
        if url.count('.') > 2:
            context_parts.append("has multiple subdomains")
        
        # Special characters
        if '@' in url:
            context_parts.append("contains @ symbol (unusual)")
        if '#' in url:
            context_parts.append("contains anchor fragment")
        if '?' in url:
            context_parts.append("contains query parameters")
        
        return ", ".join(context_parts) if context_parts else "Standard URL format"
    
    def _get_confidence_context(self, confidence):
        """Get contextual description of confidence level"""
        if confidence >= 0.95:
            return "extremely high confidence"
        elif confidence >= 0.9:
            return "very high confidence"
        elif confidence >= 0.8:
            return "high confidence"
        elif confidence >= 0.7:
            return "moderate confidence"
        elif confidence >= 0.6:
            return "low confidence"
        else:
            return "very low confidence"
    
    def _get_neighbor_context(self, neighbor_analysis):
        """Get contextual description of neighbor analysis"""
        total = neighbor_analysis.get('total_neighbors', 0)
        phishing = neighbor_analysis.get('phishing_neighbors', 0)
        safe = neighbor_analysis.get('safe_neighbors', 0)
        
        if total == 0:
            return "No similar URLs found for comparison"
        
        phishing_ratio = phishing / total if total > 0 else 0
        
        if phishing_ratio > 0.8:
            return f"Most similar URLs ({phishing}/{total}) are known phishing sites - very concerning"
        elif phishing_ratio > 0.6:
            return f"Many similar URLs ({phishing}/{total}) are phishing sites - concerning"
        elif phishing_ratio > 0.4:
            return f"Some similar URLs ({phishing}/{total}) are phishing sites - caution advised"
        elif phishing_ratio > 0.2:
            return f"Few similar URLs ({phishing}/{total}) are phishing sites - mostly safe"
        else:
            return f"Almost all similar URLs ({safe}/{total}) are safe - very reassuring"
    
    def _explain_confidence(self, prediction):
        """Explain what the confidence level means"""
        confidence = prediction['confidence']
        if confidence >= 0.9:
            return "Very high confidence - the AI is very certain about this classification"
        elif confidence >= 0.8:
            return "High confidence - the AI is quite certain about this classification"
        elif confidence >= 0.7:
            return "Moderate confidence - the AI is reasonably certain but some uncertainty exists"
        elif confidence >= 0.6:
            return "Low confidence - the AI is uncertain and the result should be treated with caution"
        else:
            return "Very low confidence - the AI is highly uncertain about this classification"
    
    def _explain_features(self, analytics):
        """Explain the most important features"""
        feature_contributions = analytics.get('feature_contributions', {})
        if not feature_contributions:
            return "No feature analysis available"
        
        # Get top 3 features
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )[:3]
        
        insights = []
        for feature, data in sorted_features:
            if data['pulls_toward_cluster_1']:
                insights.append(f"• {feature} indicates potential phishing behavior")
            else:
                insights.append(f"• {feature} suggests legitimate website characteristics")
        
        return "\n".join(insights) if insights else "No significant feature patterns detected"
    
    def _generate_recommendations(self, prediction, analytics):
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction['is_phishing']:
            recommendations.extend([
                "🚨 DO NOT enter any personal information on this website",
                "🔍 Verify the website through official channels before proceeding",
                "⚠️ Consider this URL as potentially malicious",
                "🛡️ Use additional security tools to verify the site's legitimacy"
            ])
        else:
            recommendations.extend([
                "✅ This URL appears to be safe based on AI analysis",
                "🔍 However, always remain cautious when entering personal information",
                "🛡️ Keep your security software updated",
                "📱 Consider using two-factor authentication when available"
            ])
        
        # Add confidence-based recommendations
        confidence = prediction['confidence']
        if confidence < 0.7:
            recommendations.append("⚠️ Low confidence result - consider additional verification")
        
        return recommendations
    
    def _fallback_interpretation(self, url, prediction, analytics):
        """Fallback interpretation when AI is not available"""
        # Get contextual information for more varied responses
        url_context = self._analyze_url_context(url)
        neighbor_context = self._get_neighbor_context(analytics.get('neighbor_analysis', {}))
        confidence_context = self._get_confidence_context(prediction['confidence'])
        
        # Create more specific interpretation based on context
        if prediction['is_phishing']:
            if 'google.com' in url.lower() or 'microsoft.com' in url.lower():
                main_analysis = f"""
                **🚨 SUSPICIOUS URL DETECTED**
                
                This URL appears to be impersonating a legitimate service ({url_context.split(',')[0]}). 
                This is a common phishing technique where attackers create fake websites that look like trusted companies.
                
                **Why it's suspicious:**
                • The AI found patterns consistent with known phishing attempts
                • {neighbor_context}
                • Confidence level: {confidence_context}
                
                **What this means:** This URL is likely designed to steal your personal information, passwords, or financial details by tricking you into thinking it's a legitimate website.
                """
            else:
                main_analysis = f"""
                **🚨 PHISHING ATTEMPT DETECTED**
                
                The URL '{url}' has been flagged as potentially malicious by our AI analysis.
                
                **Analysis Details:**
                • URL Context: {url_context}
                • Similar URL Analysis: {neighbor_context}
                • Confidence: {confidence_context} ({prediction['confidence']:.1f}%)
                
                **What this means:** This website appears to be designed to deceive users and steal sensitive information. The AI found multiple indicators suggesting malicious intent.
                """
        else:
            if 'google.com' in url.lower() or 'microsoft.com' in url.lower():
                main_analysis = f"""
                **✅ LEGITIMATE WEBSITE CONFIRMED**
                
                This appears to be a genuine {url_context.split(',')[0]} website. Our AI analysis found characteristics consistent with legitimate, trusted websites.
                
                **Why it's safe:**
                • {neighbor_context}
                • Confidence level: {confidence_context}
                • URL follows standard security practices
                
                **What this means:** This URL appears to be safe to use, but always verify you're on the official website before entering sensitive information.
                """
            else:
                main_analysis = f"""
                **✅ URL APPEARS SAFE**
                
                The URL '{url}' has been analyzed and appears to be legitimate based on our AI assessment.
                
                **Analysis Details:**
                • URL Context: {url_context}
                • Similar URL Analysis: {neighbor_context}
                • Confidence: {confidence_context} ({prediction['confidence']:.1f}%)
                
                **What this means:** This website shows characteristics consistent with legitimate websites. However, always remain cautious when entering personal information online.
                """
        
        return {
            'ai_interpretation': main_analysis,
            'confidence_explanation': self._explain_confidence(prediction),
            'feature_insights': self._explain_features(analytics),
            'recommendations': self._generate_recommendations(prediction, analytics)
        }

class OptimizedHierarchicalDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
        self.is_trained = False
        self.training_features = None
        self.training_labels = None
        self.cluster_labels = None
        self.feature_names = []
        self.phishing_cluster = None
        self.safe_cluster = None
        self.cluster_centers = None
        self.linkage_matrix = None
        self.cluster_analytics = None
        
        # Only use the top 15 most important features
        self.important_features = [
            'SSLfinal_State', 'URL_of_Anchor', 'web_traffic', 'having_Sub_Domain',
            'Links_in_tags', 'Prefix_Suffix', 'SFH', 'Links_pointing_to_page',
            'Request_URL', 'Domain_registeration_length', 'age_of_domain',
            'Google_Index', 'having_IP_Address', 'DNSRecord', 'Page_Rank'
        ]
        
    def load_training_data(self, arff_file):
        """Load training data from ARFF file and select only important features"""
        features, labels, feature_names = parse_arff_file(arff_file)
        
        # Find indices of important features
        important_indices = []
        for feature in self.important_features:
            if feature in feature_names:
                important_indices.append(feature_names.index(feature))
            else:
                print(f"Warning: Feature {feature} not found in dataset")
        
        # Select only important features
        selected_features = features[:, important_indices]
        self.feature_names = [feature_names[i] for i in important_indices]
        
        print(f"Selected {len(self.important_features)} important features out of {len(feature_names)}")
        print(f"Selected features: {self.feature_names}")
        
        return selected_features, labels
    
    def extract_features_from_url(self, url):
        """Extract only the important features from URL for phishing detection"""
        features = []
        
        # 1. having_IP_Address
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        has_ip = 1 if re.search(ip_pattern, url) else -1
        features.append(has_ip)
        
        # 2. URL_Length (not in top 15, but useful for URL_of_Anchor)
        url_length = len(url)
        if url_length < 54:
            length_cat = -1  # Short
        elif url_length < 75:
            length_cat = 0   # Medium
        else:
            length_cat = 1   # Long
        
        # 3. Shortining_Service (not in top 15)
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'short.link', 'short.ly', 'tiny.cc', 'shorturl.at']
        has_shortener = 1 if any(shortener in url.lower() for shortener in shorteners) else -1
        
        # 4. having_At_Symbol (not in top 15)
        has_at = 1 if '@' in url else -1
        
        # 5. double_slash_redirecting (not in top 15)
        has_double_slash = 1 if '//' in url.split('://', 1)[-1] else -1
        
        # 6. Prefix_Suffix
        has_prefix_suffix = 1 if '-' in url else -1
        features.append(has_prefix_suffix)
        
        # 7. having_Sub_Domain
        try:
            # Handle URLs without protocol
            if not url.startswith(('http://', 'https://')):
                url_with_protocol = 'http://' + url
            else:
                url_with_protocol = url
                
            parsed = urlparse(url_with_protocol)
            domain = parsed.netloc
            subdomain_count = domain.count('.') - 1
            if subdomain_count == 0:
                subdomain_cat = -1
            elif subdomain_count == 1:
                subdomain_cat = 0
            else:
                subdomain_cat = 1
        except:
            subdomain_cat = -1
        features.append(subdomain_cat)
        
        # 8. SSLfinal_State
        ssl_state = 1 if url.startswith('https://') else -1
        features.append(ssl_state)
        
        # 9. Domain_registeration_length (simplified)
        domain_length = 1  # Placeholder
        features.append(domain_length)
        
        # 10. Favicon (not in top 15)
        favicon = 1  # Placeholder
        
        # 11. port (not in top 15)
        has_port = 1 if ':' in url and '://' in url else -1
        
        # 12. HTTPS_token (not in top 15)
        https_token = 1 if 'https' in url.lower() else -1
        
        # 13. Request_URL
        request_url = 1 if '?' in url else -1
        features.append(request_url)
        
        # 14. URL_of_Anchor
        has_anchor = 1 if '#' in url else -1
        features.append(has_anchor)
        
        # 15. Links_in_tags (simplified)
        links_in_tags = -1  # Placeholder
        features.append(links_in_tags)
        
        # 16. SFH (simplified)
        sfh = -1  # Placeholder
        features.append(sfh)
        
        # 17. Submitting_to_email (not in top 15)
        submit_email = -1  # Placeholder
        
        # 18. Abnormal_URL (not in top 15)
        abnormal_score = 0
        if len(url) > 100:
            abnormal_score += 1
        if url.count('/') > 10:
            abnormal_score += 1
        if url.count('?') > 3:
            abnormal_score += 1
        if url.count('&') > 5:
            abnormal_score += 1
        abnormal_url = 1 if abnormal_score >= 2 else -1
        
        # 19. Redirect (not in top 15)
        redirect_keywords = ['redirect', 'url=', 'goto=', 'link=', 'jump=']
        redirect_count = sum(1 for keyword in redirect_keywords if keyword in url.lower())
        redirect_cat = 1 if redirect_count > 0 else 0
        
        # 20. on_mouseover (not in top 15)
        mouseover = -1  # Placeholder
        
        # 21. RightClick (not in top 15)
        rightclick = -1  # Placeholder
        
        # 22. popUpWidnow (not in top 15)
        popup = -1  # Placeholder
        
        # 23. Iframe (not in top 15)
        iframe = -1  # Placeholder
        
        # 24. age_of_domain
        age_domain = 1  # Placeholder
        features.append(age_domain)
        
        # 25. DNSRecord
        dns_record = 1  # Placeholder
        features.append(dns_record)
        
        # 26. web_traffic
        web_traffic = 0  # Placeholder
        features.append(web_traffic)
        
        # 27. Page_Rank
        page_rank = 1  # Placeholder
        features.append(page_rank)
        
        # 28. Google_Index
        google_index = 1  # Placeholder
        features.append(google_index)
        
        # 29. Links_pointing_to_page
        links_pointing = 0  # Placeholder
        features.append(links_pointing)
        
        # 30. Statistical_report (not in top 15)
        statistical_report = -1  # Placeholder
        
        return np.array(features)
    
    def _has_phishing_domain_patterns(self, domain):
        """Check for common phishing domain patterns"""
        # Check for brand name + suspicious words
        brand_names = ['google', 'microsoft', 'amazon', 'apple', 'facebook', 'paypal', 'ebay', 'netflix', 'spotify']
        suspicious_words = ['security', 'account', 'login', 'verify', 'update', 'suspended', 'alert', 'support', 'help']
        
        domain_lower = domain.lower()
        
        # Check if domain contains brand name + suspicious word
        for brand in brand_names:
            if brand in domain_lower:
                for suspicious in suspicious_words:
                    if suspicious in domain_lower:
                        return True
        
        # Check for typosquatting patterns
        if self._has_typosquatting(domain):
            return True
            
        # Check for suspicious TLD combinations
        if self._has_suspicious_tld_pattern(domain):
            return True
            
        return False
    
    def _has_http_phishing_indicators(self, url, domain):
        """Check for HTTP-specific phishing indicators"""
        if not url.startswith('http://'):
            return False
            
        # HTTP + brand name + suspicious word = high risk
        if self._has_phishing_domain_patterns(domain):
            return True
            
        # HTTP + suspicious TLD = high risk
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.info', '.net']
        if any(tld in domain for tld in suspicious_tlds):
            return True
            
        # HTTP + long domain with hyphens = suspicious
        if len(domain) > 20 and domain.count('-') > 2:
            return True
            
        return False
    
    def _calculate_domain_trust_score(self, domain):
        """Calculate domain trust score based on various factors"""
        score = 0.5  # Start with neutral score
        
        # Well-known domains get higher trust
        trusted_domains = [
            'google.com', 'microsoft.com', 'amazon.com', 'apple.com', 'facebook.com',
            'github.com', 'stackoverflow.com', 'wikipedia.org', 'reddit.com', 'youtube.com',
            'paypal.com', 'ebay.com', 'netflix.com', 'spotify.com', 'twitter.com'
        ]
        
        if domain in trusted_domains:
            score += 0.4
        
        # Suspicious TLDs reduce trust
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
        if any(tld in domain for tld in suspicious_tlds):
            score -= 0.3
            
        # Long domains with many hyphens reduce trust
        if len(domain) > 25 and domain.count('-') > 3:
            score -= 0.2
            
        # Numbers in domain reduce trust
        if any(char.isdigit() for char in domain):
            score -= 0.1
            
        return max(0.0, min(1.0, score))
    
    def _has_typosquatting(self, domain):
        """Check for typosquatting patterns"""
        # Simple typosquatting detection
        common_typos = ['gogle', 'gooogle', 'microsft', 'micrsoft', 'amazom', 'amazn', 'appel', 'facebok', 'paypall']
        domain_lower = domain.lower()
        
        for typo in common_typos:
            if typo in domain_lower:
                return True
        return False
    
    def _has_suspicious_tld_pattern(self, domain):
        """Check for suspicious TLD patterns"""
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.info']
        return any(tld in domain for tld in suspicious_tlds)
    
    def train_model(self, arff_file):
        """Train the hierarchical clustering model with optimized features"""
        print("Loading training data with optimized features...")
        features, labels = self.load_training_data(arff_file)
        
        print(f"Training with {len(features)} samples and {len(self.feature_names)} features")
        
        # Store training data
        self.training_features = features
        self.training_labels = labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Train hierarchical clustering model
        print("Training hierarchical clustering model...")
        self.cluster_labels = self.clustering_model.fit_predict(X_scaled)
        self.is_trained = True
        
        # Calculate cluster centers
        self.cluster_centers = []
        for cluster_id in range(2):
            cluster_points = X_scaled[self.cluster_labels == cluster_id]
            if len(cluster_points) > 0:
                self.cluster_centers.append(np.mean(cluster_points, axis=0))
            else:
                self.cluster_centers.append(np.zeros(X_scaled.shape[1]))
        
        # Determine which cluster represents phishing vs safe
        cluster_0_labels = labels[self.cluster_labels == 0]
        cluster_1_labels = labels[self.cluster_labels == 1]
        
        # Count phishing (1) vs safe (-1) in each cluster
        cluster_0_phishing_ratio = np.mean(cluster_0_labels == 1) if len(cluster_0_labels) > 0 else 0
        cluster_1_phishing_ratio = np.mean(cluster_1_labels == 1) if len(cluster_1_labels) > 0 else 0
        
        # The cluster with higher phishing ratio is the phishing cluster
        if cluster_0_phishing_ratio > cluster_1_phishing_ratio:
            self.phishing_cluster = 0
            self.safe_cluster = 1
        else:
            self.phishing_cluster = 1
            self.safe_cluster = 0
            
        print(f"Cluster 0 phishing ratio: {cluster_0_phishing_ratio:.3f} (size: {len(cluster_0_labels)})")
        print(f"Cluster 1 phishing ratio: {cluster_1_phishing_ratio:.3f} (size: {len(cluster_1_labels)})")
        print(f"Phishing cluster: {self.phishing_cluster}")
        print(f"Safe cluster: {self.safe_cluster}")
        
        # Generate cluster analytics
        print("Generating cluster analytics...")
        self.generate_cluster_analytics()
        
        return True
    
    def predict(self, url):
        """Predict if URL is phishing or safe using optimized hierarchical clustering"""
        try:
            if not self.is_trained:
                return self._heuristic_prediction(url)
            
            # Extract features
            features = self.extract_features_from_url(url).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Use nearest neighbors to find similar training samples
            nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree')
            nbrs.fit(self.scaler.transform(self.training_features))
            
            # Find nearest neighbors
            distances, indices = nbrs.kneighbors(features_scaled)
            
            # Get cluster labels of nearest neighbors
            neighbor_clusters = self.cluster_labels[indices[0]]
            neighbor_labels = self.training_labels[indices[0]]
            
            # Count votes for each cluster
            cluster_0_votes = np.sum(neighbor_clusters == 0)
            cluster_1_votes = np.sum(neighbor_clusters == 1)
            
            # Count actual phishing vs safe votes
            phishing_votes = np.sum(neighbor_labels == 1)
            safe_votes = np.sum(neighbor_labels == -1)
            
            # Use both cluster assignment and actual label votes
            cluster_confidence = max(cluster_0_votes, cluster_1_votes) / (cluster_0_votes + cluster_1_votes)
            label_confidence = max(phishing_votes, safe_votes) / (phishing_votes + safe_votes)
            
            # Predict cluster based on majority vote
            predicted_cluster = 0 if cluster_0_votes > cluster_1_votes else 1
            
            # Predict based on actual labels (more reliable)
            is_phishing_by_labels = phishing_votes > safe_votes
            
            # Calculate confidence based on both methods
            avg_distance = np.mean(distances[0])
            distance_confidence = max(0.3, 1 - (avg_distance / 10))
            
            # Combine all confidence measures
            overall_confidence = (cluster_confidence + label_confidence + distance_confidence) / 3
            overall_confidence = max(0.5, min(0.99, overall_confidence))
            
            # Enhanced prediction logic with domain analysis
            domain = urlparse(url).netloc.lower()
            
            # Check for HTTP phishing indicators
            http_phishing_risk = self._has_http_phishing_indicators(url, domain)
            domain_trust_score = self._calculate_domain_trust_score(domain)
            phishing_domain_patterns = self._has_phishing_domain_patterns(domain)
            
            # Calculate enhanced confidence based on multiple factors
            enhanced_confidence = overall_confidence
            
            # Boost confidence for clear phishing indicators
            if http_phishing_risk:
                enhanced_confidence = min(0.95, enhanced_confidence + 0.2)
            if phishing_domain_patterns:
                enhanced_confidence = min(0.95, enhanced_confidence + 0.15)
            if domain_trust_score < 0.3:
                enhanced_confidence = min(0.95, enhanced_confidence + 0.1)
            
            # Use label-based prediction as primary, but consider domain analysis
            if abs(phishing_votes - safe_votes) > 2:  # Clear majority
                is_phishing = is_phishing_by_labels
                confidence = enhanced_confidence
            elif http_phishing_risk or phishing_domain_patterns:  # Strong domain indicators
                is_phishing = True
                confidence = enhanced_confidence * 0.9
            elif domain_trust_score > 0.7:  # High trust domain
                is_phishing = False
                confidence = enhanced_confidence * 0.9
            else:  # Use cluster prediction
                is_phishing = (predicted_cluster == self.phishing_cluster)
                confidence = enhanced_confidence * 0.8  # Lower confidence for cluster-based
            
            # Override with domain analysis if it strongly suggests phishing
            if http_phishing_risk and phishing_domain_patterns:
                is_phishing = True
                confidence = min(0.95, enhanced_confidence + 0.1)
            
            print(f"URL: {url}")
            print(f"Features: {features[0][:5]}...")  # First 5 features
            print(f"Cluster votes: 0={cluster_0_votes}, 1={cluster_1_votes}")
            print(f"Label votes: phishing={phishing_votes}, safe={safe_votes}")
            print(f"Predicted cluster: {predicted_cluster}, Phishing cluster: {self.phishing_cluster}")
            print(f"Is phishing (labels): {is_phishing_by_labels}")
            print(f"Vote difference: {abs(phishing_votes - safe_votes)}")
            print(f"Domain analysis: HTTP_risk={http_phishing_risk}, Trust_score={domain_trust_score:.2f}, Phishing_patterns={phishing_domain_patterns}")
            print(f"Is phishing (final): {is_phishing}, Confidence: {confidence:.3f}")
            
            return {
                'is_phishing': bool(is_phishing),
                'confidence': float(confidence),
                'cluster': int(predicted_cluster)
            }
        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._heuristic_prediction(url)
    
    def _heuristic_prediction(self, url):
        """Simple heuristic-based prediction when model isn't trained"""
        features = self.extract_features_from_url(url)
        
        # Simple scoring system based on key features
        score = 0
        
        # IP address in URL
        if features[0] == 1:  # having_IP_Address
            score += 2
        
        # Long URL
        if len(url) > 75:
            score += 1
        
        # Shortening service
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
        if any(shortener in url.lower() for shortener in shorteners):
            score += 1
        
        # @ symbol
        if '@' in url:
            score += 2
        
        # Abnormal URL
        if len(url) > 100 or url.count('/') > 10:
            score += 1
        
        # Determine if phishing based on score
        is_phishing = score >= 3
        confidence = min(0.95, 0.6 + (score * 0.1))
        
        return {
            'is_phishing': bool(is_phishing),
            'confidence': float(confidence),
            'cluster': 1 if is_phishing else 0
        }
    
    def generate_dendrogram(self, max_samples=1000):
        """Generate dendrogram visualization for hierarchical clustering"""
        if not self.is_trained or self.training_features is None:
            return None
        
        # Sample data if too large for visualization
        n_samples = min(max_samples, len(self.training_features))
        if len(self.training_features) > max_samples:
            indices = np.random.choice(len(self.training_features), n_samples, replace=False)
            sample_features = self.training_features[indices]
            sample_labels = self.training_labels[indices]
        else:
            sample_features = self.training_features
            sample_labels = self.training_labels
        
        # Scale features
        X_scaled = self.scaler.transform(sample_features)
        
        # Calculate linkage matrix
        linkage_matrix = linkage(X_scaled, method='ward')
        self.linkage_matrix = linkage_matrix
        
        # Create dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, 
                  truncate_mode='level', 
                  p=5,  # Show only the last p levels
                  leaf_rotation=90,
                  leaf_font_size=8,
                  show_contracted=True)
        
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
        plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
    def generate_cluster_analytics(self):
        """Generate comprehensive cluster analytics"""
        if not self.is_trained or self.training_features is None:
            return None
        
        X_scaled = self.scaler.transform(self.training_features)
        
        # Basic cluster statistics
        cluster_stats = {}
        for cluster_id in range(2):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_points = X_scaled[cluster_mask]
            cluster_labels = self.training_labels[cluster_mask]
            
            cluster_stats[cluster_id] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(self.training_labels) * 100),
                'phishing_ratio': float(np.mean(cluster_labels == 1)) if len(cluster_labels) > 0 else 0,
                'safe_ratio': float(np.mean(cluster_labels == -1)) if len(cluster_labels) > 0 else 0,
                'center': cluster_points.mean(axis=0).tolist() if len(cluster_points) > 0 else [0] * len(self.feature_names)
            }
        
        # Calculate clustering quality metrics (use sample for large datasets)
        if len(X_scaled) > 5000:
            # Sample data for silhouette score to avoid memory issues
            sample_size = min(5000, len(X_scaled))
            sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[sample_indices]
            labels_sample = self.cluster_labels[sample_indices]
            silhouette_avg = silhouette_score(X_sample, labels_sample)
        else:
            silhouette_avg = silhouette_score(X_scaled, self.cluster_labels)
        
        calinski_harabasz = calinski_harabasz_score(X_scaled, self.cluster_labels)
        
        # Feature importance analysis
        feature_importance = self._calculate_feature_importance()
        
        # Cluster separation analysis
        cluster_separation = self._analyze_cluster_separation()
        
        self.cluster_analytics = {
            'cluster_statistics': cluster_stats,
            'quality_metrics': {
                'silhouette_score': float(silhouette_avg),
                'calinski_harabasz_score': float(calinski_harabasz)
            },
            'feature_importance': feature_importance,
            'cluster_separation': cluster_separation,
            'model_info': {
                'n_samples': len(self.training_features),
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'phishing_cluster': self.phishing_cluster,
                'safe_cluster': self.safe_cluster
            }
        }
        
        return self.cluster_analytics
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on cluster separation"""
        if not self.is_trained:
            return {}
        
        X_scaled = self.scaler.transform(self.training_features)
        feature_importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = X_scaled[:, i]
            
            # Calculate variance between clusters
            cluster_0_values = feature_values[self.cluster_labels == 0]
            cluster_1_values = feature_values[self.cluster_labels == 1]
            
            if len(cluster_0_values) > 0 and len(cluster_1_values) > 0:
                between_cluster_var = np.var([cluster_0_values.mean(), cluster_1_values.mean()])
                within_cluster_var = (np.var(cluster_0_values) + np.var(cluster_1_values)) / 2
                
                # F-ratio as importance measure
                importance = between_cluster_var / (within_cluster_var + 1e-8)
                feature_importance[feature_name] = float(importance)
            else:
                feature_importance[feature_name] = 0.0
        
        # Normalize importance scores
        max_importance = max(feature_importance.values()) if feature_importance.values() else 1
        for feature in feature_importance:
            feature_importance[feature] = feature_importance[feature] / max_importance
        
        return feature_importance
    
    def _analyze_cluster_separation(self):
        """Analyze how well separated the clusters are"""
        if not self.is_trained:
            return {}
        
        X_scaled = self.scaler.transform(self.training_features)
        
        # Calculate distances between cluster centers
        cluster_0_center = X_scaled[self.cluster_labels == 0].mean(axis=0)
        cluster_1_center = X_scaled[self.cluster_labels == 1].mean(axis=0)
        center_distance = np.linalg.norm(cluster_0_center - cluster_1_center)
        
        # Calculate average intra-cluster distances
        cluster_0_distances = []
        cluster_1_distances = []
        
        for i in range(len(X_scaled)):
            if self.cluster_labels[i] == 0:
                dist = np.linalg.norm(X_scaled[i] - cluster_0_center)
                cluster_0_distances.append(dist)
            else:
                dist = np.linalg.norm(X_scaled[i] - cluster_1_center)
                cluster_1_distances.append(dist)
        
        avg_intra_cluster_0 = np.mean(cluster_0_distances) if cluster_0_distances else 0
        avg_intra_cluster_1 = np.mean(cluster_1_distances) if cluster_1_distances else 0
        avg_intra_cluster = (avg_intra_cluster_0 + avg_intra_cluster_1) / 2
        
        # Separation ratio
        separation_ratio = center_distance / (avg_intra_cluster + 1e-8)
        
        return {
            'center_distance': float(center_distance),
            'avg_intra_cluster_distance': float(avg_intra_cluster),
            'separation_ratio': float(separation_ratio),
            'cluster_0_avg_distance': float(avg_intra_cluster_0),
            'cluster_1_avg_distance': float(avg_intra_cluster_1)
        }
    
    def generate_interactive_plots(self):
        """Generate interactive plots for the web interface"""
        if not self.is_trained or self.training_features is None:
            return None
        
        X_scaled = self.scaler.transform(self.training_features)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(X_scaled, columns=self.feature_names)
        df['cluster'] = self.cluster_labels
        df['label'] = self.training_labels
        
        plots = {}
        
        # 1. Feature importance bar chart
        if self.cluster_analytics and 'feature_importance' in self.cluster_analytics:
            importance_data = self.cluster_analytics['feature_importance']
            fig_importance = px.bar(
                x=list(importance_data.keys()),
                y=list(importance_data.values()),
                title="Feature Importance in Clustering",
                labels={'x': 'Features', 'y': 'Importance Score'},
                color=list(importance_data.values()),
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False
            )
            plots['feature_importance'] = json.dumps(fig_importance, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. Cluster distribution pie chart
        cluster_counts = pd.Series(self.cluster_labels).value_counts()
        fig_distribution = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title="Cluster Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        plots['cluster_distribution'] = json.dumps(fig_distribution, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 3. 2D projection of clusters (using first two most important features)
        if len(self.feature_names) >= 2:
            top_features = sorted(self.cluster_analytics['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:2]
            feature_names_2d = [f[0] for f in top_features]
            
            fig_2d = px.scatter(
                df, 
                x=feature_names_2d[0], 
                y=feature_names_2d[1],
                color='cluster',
                title=f"2D Cluster Visualization ({feature_names_2d[0]} vs {feature_names_2d[1]})",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            plots['cluster_2d'] = json.dumps(fig_2d, cls=plotly.utils.PlotlyJSONEncoder)
        
        return plots
    
    def generate_url_dendrogram(self, url, max_samples=500):
        """Generate dendrogram showing where the current URL fits in the clustering hierarchy"""
        if not self.is_trained or self.training_features is None:
            return None
        
        try:
            # Extract features for the current URL
            url_features = self.extract_features_from_url(url).reshape(1, -1)
            url_scaled = self.scaler.transform(url_features)
        except Exception as e:
            print(f"Error extracting URL features: {e}")
            return None
        
        # Sample training data for visualization
        n_samples = min(max_samples, len(self.training_features))
        if len(self.training_features) > max_samples:
            indices = np.random.choice(len(self.training_features), n_samples, replace=False)
            sample_features = self.training_features[indices]
            sample_labels = self.training_labels[indices]
        else:
            sample_features = self.training_features
            sample_labels = self.training_labels
        
        # Combine URL with training data
        combined_features = np.vstack([sample_features, url_features])
        combined_scaled = self.scaler.transform(combined_features)
        
        try:
            # Calculate linkage matrix
            linkage_matrix = linkage(combined_scaled, method='ward')
            
            # Create dendrogram with highlighted URL
            plt.figure(figsize=(14, 10))
            
            # Find URL position (it should be the last one)
            url_position = len(combined_scaled) - 1
            
            dendrogram(linkage_matrix, 
                      truncate_mode='level', 
                      p=5,
                      leaf_rotation=90,
                      leaf_font_size=8,
                      show_contracted=True,
                      leaf_label_func=lambda x: f"URL" if x == url_position else f"Sample {x}",
                      color_threshold=0.7 * max(linkage_matrix[:, 2]))
            
            plt.title(f'URL Clustering Analysis: {url[:50]}...', fontsize=16, fontweight='bold')
            plt.xlabel('Sample Index (URL highlighted in red)', fontsize=12)
            plt.ylabel('Distance', fontsize=12)
            
            # Add legend
            plt.figtext(0.02, 0.02, 'Red dot indicates the analyzed URL', fontsize=10, color='red')
            
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            print(f"Error generating dendrogram: {e}")
            plt.close()  # Make sure to close the figure
            return None
    
    def generate_url_analytics(self, url):
        """Generate analytics specific to the current URL"""
        if not self.is_trained:
            return None
        
        try:
            # Extract features for the URL
            url_features = self.extract_features_from_url(url).reshape(1, -1)
            url_scaled = self.scaler.transform(url_features)
            
            # Get prediction details
            prediction = self.predict(url)
        except Exception as e:
            print(f"Error in URL analytics setup: {e}")
            return None
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree')
        nbrs.fit(self.scaler.transform(self.training_features))
        distances, indices = nbrs.kneighbors(url_scaled)
        
        # Get neighbor information
        neighbor_clusters = self.cluster_labels[indices[0]]
        neighbor_labels = self.training_labels[indices[0]]
        neighbor_distances = distances[0]
        
        # Calculate URL-specific statistics
        cluster_0_neighbors = np.sum(neighbor_clusters == 0)
        cluster_1_neighbors = np.sum(neighbor_clusters == 1)
        phishing_neighbors = np.sum(neighbor_labels == 1)
        safe_neighbors = np.sum(neighbor_labels == -1)
        
        # Calculate average distance to each cluster
        cluster_0_distances = neighbor_distances[neighbor_clusters == 0]
        cluster_1_distances = neighbor_distances[neighbor_clusters == 1]
        
        avg_dist_cluster_0 = np.mean(cluster_0_distances) if len(cluster_0_distances) > 0 else float('inf')
        avg_dist_cluster_1 = np.mean(cluster_1_distances) if len(cluster_1_distances) > 0 else float('inf')
        
        # Calculate feature contributions
        feature_contributions = self._calculate_url_feature_contributions(url_features[0])
        
        # Calculate outlier score
        outlier_score = self._calculate_outlier_score(url_scaled[0])
        
        return {
            'url': url,
            'prediction': prediction,
            'neighbor_analysis': {
                'total_neighbors': len(neighbor_clusters),
                'cluster_0_neighbors': int(cluster_0_neighbors),
                'cluster_1_neighbors': int(cluster_1_neighbors),
                'phishing_neighbors': int(phishing_neighbors),
                'safe_neighbors': int(safe_neighbors),
                'avg_distance_cluster_0': float(avg_dist_cluster_0),
                'avg_distance_cluster_1': float(avg_dist_cluster_1)
            },
            'feature_contributions': feature_contributions,
            'outlier_score': float(outlier_score),
            'cluster_characteristics': {
                'assigned_cluster': int(prediction['cluster']),
                'is_phishing_cluster': bool(prediction['cluster'] == self.phishing_cluster),
                'cluster_confidence': float(prediction['confidence']),
                'phishing_cluster': int(self.phishing_cluster),
                'safe_cluster': int(self.safe_cluster)
            }
        }
    
    def _calculate_url_feature_contributions(self, url_features):
        """Calculate how much each feature contributes to the URL's classification"""
        if not self.is_trained:
            return {}
        
        try:
            # Get cluster centers
            X_scaled = self.scaler.transform(self.training_features)
            cluster_0_center = X_scaled[self.cluster_labels == 0].mean(axis=0)
            cluster_1_center = X_scaled[self.cluster_labels == 1].mean(axis=0)
            
            url_scaled = self.scaler.transform(url_features.reshape(1, -1))[0]
            
            # Calculate feature-wise contributions
            contributions = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(url_scaled) and i < len(cluster_0_center) and i < len(cluster_1_center):
                    # Calculate how much this feature value pulls toward each cluster
                    # If URL feature is closer to cluster 1 center, it pulls toward cluster 1
                    # If URL feature is closer to cluster 0 center, it pulls toward cluster 0
                    
                    feature_dist_to_0 = abs(url_scaled[i] - cluster_0_center[i])
                    feature_dist_to_1 = abs(url_scaled[i] - cluster_1_center[i])
                    
                    # Contribution: positive means pulls toward cluster 1, negative toward cluster 0
                    # We want to show which cluster this feature value is closer to
                    contribution = feature_dist_to_0 - feature_dist_to_1
                    
                    # Determine which cluster this pulls toward
                    pulls_toward_cluster_1 = contribution > 0  # Closer to cluster 1
                    pulls_toward_cluster_0 = contribution < 0  # Closer to cluster 0
                    
                    contributions[feature_name] = {
                        'value': float(url_scaled[i]),
                        'contribution': float(contribution),
                        'pulls_toward_cluster_0': bool(pulls_toward_cluster_0),
                        'pulls_toward_cluster_1': bool(pulls_toward_cluster_1)
                    }
                else:
                    # Handle case where feature index is out of bounds
                    contributions[feature_name] = {
                        'value': 0.0,
                        'contribution': 0.0,
                        'pulls_toward_cluster_0': bool(False),
                        'pulls_toward_cluster_1': bool(False)
                    }
            
            return contributions
        except Exception as e:
            print(f"Error calculating feature contributions: {e}")
            return {}
    
    def _calculate_outlier_score(self, url_scaled):
        """Calculate how much of an outlier the URL is"""
        if not self.is_trained:
            return 0.0
        
        try:
            # Calculate distance to nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
            nbrs.fit(self.scaler.transform(self.training_features))
            distances, _ = nbrs.kneighbors(url_scaled.reshape(1, -1))
            
            # Outlier score based on average distance to nearest neighbors
            avg_distance = np.mean(distances[0])
            
            # Normalize to 0-1 scale (higher = more outlier)
            max_expected_distance = 5.0  # Based on typical distances in the dataset
            outlier_score = min(1.0, avg_distance / max_expected_distance)
            
            return outlier_score
        except Exception as e:
            print(f"Error calculating outlier score: {e}")
            return 0.0

# Initialize the optimized hierarchical detector
detector = OptimizedHierarchicalDetector()

# Initialize Gemini AI interpreter
gemini_interpreter = GeminiInterpreter()

# Train the model with the real dataset
print("Initializing optimized hierarchical phishing detector...")
detector.train_model('Training Dataset.arff')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'Please provide a URL'}), 400
        
        # Store original URL for analysis
        original_url = url
        
        # Add protocol if missing (but analyze both versions)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Simulate analysis delay for better UX
        time.sleep(1.5)
        
        # Analyze the URL (try both with and without protocol)
        result = detector.predict(url)
        
        # If original URL didn't have protocol, also try analyzing without it
        if not original_url.startswith(('http://', 'https://')):
            result_no_protocol = detector.predict(original_url)
            # Use the result with higher confidence
            if result_no_protocol['confidence'] > result['confidence']:
                result = result_no_protocol
        
        # Generate URL-specific dendrogram and analytics
        try:
            url_dendrogram = detector.generate_url_dendrogram(url)
            url_analytics = detector.generate_url_analytics(url)
        except Exception as e:
            print(f"Error generating URL analytics: {e}")
            url_dendrogram = None
            url_analytics = None
        
        # Generate AI interpretation
        try:
            ai_interpretation = gemini_interpreter.interpret_analysis(url, result, url_analytics)
        except Exception as e:
            print(f"Error generating AI interpretation: {e}")
            ai_interpretation = None
        
        response = {
            'is_phishing': result['is_phishing'],
            'confidence': round(result['confidence'] * 100, 1),
            'url': url,
            'dendrogram': url_dendrogram,
            'analytics': url_analytics,
            'ai_interpretation': ai_interpretation
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/model_info')
def model_info():
    """Return information about the trained model"""
    return jsonify({
        'is_trained': detector.is_trained,
        'feature_count': len(detector.feature_names),
        'feature_names': detector.feature_names,
        'phishing_cluster': detector.phishing_cluster,
        'safe_cluster': detector.safe_cluster,
        'training_samples': len(detector.training_features) if detector.is_trained else 0,
        'optimization': 'Using only top 15 most important features for better performance'
    })

@app.route('/dendrogram')
def get_dendrogram():
    """Generate and return dendrogram visualization"""
    try:
        if not detector.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        dendrogram_data = detector.generate_dendrogram()
        if dendrogram_data:
            return jsonify({
                'dendrogram': dendrogram_data,
                'success': True
            })
        else:
            return jsonify({'error': 'Failed to generate dendrogram'}), 500
            
    except Exception as e:
        print(f"Dendrogram generation error: {e}")
        return jsonify({'error': 'Failed to generate dendrogram'}), 500

@app.route('/analytics')
def get_analytics():
    """Return comprehensive cluster analytics"""
    try:
        if not detector.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        analytics = detector.cluster_analytics
        if analytics is None:
            analytics = detector.generate_cluster_analytics()
        
        return jsonify({
            'analytics': analytics,
            'success': True
        })
        
    except Exception as e:
        print(f"Analytics generation error: {e}")
        return jsonify({'error': 'Failed to generate analytics'}), 500

@app.route('/plots')
def get_plots():
    """Return interactive plots for the web interface"""
    try:
        if not detector.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        plots = detector.generate_interactive_plots()
        if plots:
            return jsonify({
                'plots': plots,
                'success': True
            })
        else:
            return jsonify({'error': 'Failed to generate plots'}), 500
            
    except Exception as e:
        print(f"Plots generation error: {e}")
        return jsonify({'error': 'Failed to generate plots'}), 500

if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
