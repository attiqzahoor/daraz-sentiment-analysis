from fastapi import FastAPI, HTTPException, Query
import requests
import re
import json
from collections import defaultdict
from transformers import pipeline
import uvicorn
from typing import List, Dict, Any

app = FastAPI()

# Initialize sentiment analyzer
sentiment_analyzer = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=-1,
    truncation=True
)

def get_daraz_reviews(url: str, max_pages: int = 3) -> List[Dict[str, Any]]:
    """Fetch reviews from Daraz product URL"""
    product_id = re.search(r'-i(\d+)', url)
    if not product_id:
        return []
    
    product_id = product_id.group(1)
    reviews = []
    
    for page in range(1, max_pages + 1):
        api_url = f"https://my.daraz.pk/pdp/review/getReviewList?itemId={product_id}&pageSize=20&filter=0&sort=0&pageNo={page}"
        try:
            response = requests.get(api_url, timeout=10)
            data = response.json()
            items = data.get('model', {}).get('items', [])
            
            for item in items:
                reviews.append({
                    'author': item.get('buyerName'),
                    'rating': item.get('rating'),
                    'date': item.get('reviewTime'),
                    'content': item.get('reviewContent'),
                    'likes': item.get('likeCount')
                })
                
            if not items:
                break
                
        except Exception as e:
            print(f"Error fetching page {page}: {str(e)}")
            continue
            
    return reviews

def analyze_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze reviews and return sentiment analysis"""
    if not reviews:
        return {
            'stats': {'total': 0},
            'issues': {},
            'time_taken': 0
        }

    # Sentiment analysis
    sentiments = []
    for review in reviews:
        try:
            content = review['content'][:512]
            result = sentiment_analyzer(content)[0]
            sentiments.append(result['label'])
        except:
            sentiments.append('NEUTRAL')

    pos = sum(1 for s in sentiments if s == 'POSITIVE')
    neg = sum(1 for s in sentiments if s == 'NEGATIVE')
    neu = len(reviews) - pos - neg

    # Common issues in negative reviews
    negative_reviews = [r for r, s in zip(reviews, sentiments) if s == 'NEGATIVE']
    issue_counts = defaultdict(int)
    
    issue_keywords = {
        'quality': ['quality', 'poor', 'cheap', 'broken', 'defective'],
        'delivery': ['delivery', 'late', 'slow', 'shipping'],
        'price': ['price', 'expensive', 'overpriced'],
        'description': ['description', 'different', 'wrong'],
        'service': ['service', 'customer', 'support']
    }

    for review in negative_reviews:
        content = review['content'].lower()
        for issue, keywords in issue_keywords.items():
            if any(keyword in content for keyword in keywords):
                issue_counts[issue] += 1
                break

    return {
        'stats': {'total': len(reviews), 'positive': pos, 'negative': neg, 'neutral': neu},
        'issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    }

@app.get("/analyze")
async def analyze_product(
    url: str = Query(..., description="Daraz product URL"),
    max_pages: int = Query(1, description="Pages to scrape (1-3)", ge=1, le=3)
) -> Dict[str, Any]:
    """
    Analyze Daraz product reviews
    - Provide product URL
    - Returns sentiment analysis and common issues
    """
    try:
        if "daraz.pk" not in url:
            raise HTTPException(status_code=400, detail="Invalid Daraz URL")
        
        print(f"Fetching reviews from: {url}")
        reviews = get_daraz_reviews(url, max_pages)
        
        if not reviews:
            return {
                "status": "success",
                "message": "No reviews found",
                "data": {
                    "reviews_count": 0,
                    "analysis": None
                }
            }

        analysis = analyze_reviews(reviews)
        
        return {
            "status": "success",
            "data": {
                "reviews_count": len(reviews),
                "sentiment": {
                    "positive": analysis['stats']['positive'],
                    "negative": analysis['stats']['negative'],
                    "neutral": analysis['stats']['neutral'],
                    "positive_percent": f"{analysis['stats']['positive']/len(reviews):.1%}",
                    "negative_percent": f"{analysis['stats']['negative']/len(reviews):.1%}"
                },
                "common_issues": [
                    {"issue": k, "count": v} 
                    for k, v in analysis['issues'].items()
                ],
                "sample_reviews": reviews[:3]  # Return first 3 reviews as sample
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)