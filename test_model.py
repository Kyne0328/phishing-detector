#!/usr/bin/env python3
"""
Test the phishing detector model with various URLs
"""

import requests
import time

def test_urls():
    """Test various URLs to see the model's behavior"""
    
    test_cases = [
        # Known safe URLs
        ("https://www.google.com", "Should be SAFE"),
        ("https://www.github.com", "Should be SAFE"),
        ("https://www.microsoft.com", "Should be SAFE"),
        ("https://www.stackoverflow.com", "Should be SAFE"),
        
        # Known suspicious URLs
        ("https://phishing-site.tk/login", "Should be PHISHING"),
        ("https://bit.ly/suspicious-link", "Should be PHISHING"),
        ("https://192.168.1.1/fake-bank", "Should be PHISHING"),
        ("https://fake-paypal-verification.com", "Should be PHISHING"),
        
        # Edge cases
        ("https://very-long-suspicious-url-with-many-special-characters-and-numbers-12345.com", "Should be PHISHING"),
        ("https://short.ly/abc", "Should be PHISHING"),
    ]
    
    print("Testing Phishing Detector Model")
    print("=" * 60)
    
    for url, expected in test_cases:
        try:
            print(f"\nTesting: {url}")
            print(f"Expected: {expected}")
            
            response = requests.post(
                'http://localhost:5000/analyze',
                json={'url': url},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                result = "PHISHING" if data['is_phishing'] else "SAFE"
                confidence = data['confidence']
                
                print(f"Result: {result}")
                print(f"Confidence: {confidence}%")
                
                # Check if result matches expectation
                if "SAFE" in expected and not data['is_phishing']:
                    print("✅ CORRECT")
                elif "PHISHING" in expected and data['is_phishing']:
                    print("✅ CORRECT")
                else:
                    print("❌ INCORRECT")
            else:
                print(f"Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_urls()

