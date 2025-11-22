"""Test script to verify neutral sentiment classification."""
import sys
sys.path.insert(0, 'c:/Users/User/Documents/multimodal-product-analyzer')

from src.models.pretrained_fusion import PretrainedFusionModel
from transformers import pipeline, CLIPProcessor, CLIPModel

print("Loading models...")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Creating fusion model...")
categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Books', 'Toys', 'Beauty', 'Automotive', 'Food', 'Health']
sentiments = ['negative', 'neutral', 'positive']
fusion_model = PretrainedFusionModel(clip_model, clip_processor, sentiment_pipeline, categories, sentiments)

# Test neutral reviews
test_reviews = [
    "The phone is okay. Display is decent and battery lasts through the day. However, camera is average and there is bloatware. Overall acceptable but nothing special.",
    "I recently used the Oppo Reno 12 Pro 5G, and overall it's a pretty balanced device. On the plus side, it has a 120 Hz AMOLED display. However, it's not perfect. There's no wireless charging.",
    "The product is fine. It works as expected but nothing exceptional. Decent quality for the price.",
]

print("\n" + "="*80)
print("TESTING NEUTRAL SENTIMENT CLASSIFICATION")
print("="*80)

for i, review in enumerate(test_reviews, 1):
    print(f"\n📝 Test {i}")
    print(f"Review: {review[:100]}...")
    
    analysis = fusion_model._analyze_sentiment_comprehensive(review)
    
    star_rating = analysis['final_star_rating']
    if star_rating >= 4:
        sentiment = 'positive'
    elif star_rating == 3:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    
    print(f"\n✓ Sentiment: {sentiment.upper()}")
    print(f"  Star Rating: {star_rating}/5")
    print(f"  Confidence: {analysis['confidence']:.2%}")
    print(f"  Rule Triggered: {analysis.get('rule_triggered', 'N/A')}")
    print(f"  Scores - Pos: {analysis['positive_score']:.2f}, Neg: {analysis['negative_score']:.2f}, Neu: {analysis['neutral_score']:.2f}")
    
    if sentiment == 'neutral':
        print("  ✅ PASSED - Classified as neutral")
    else:
        print(f"  ❌ FAILED - Classified as {sentiment} instead of neutral")

print("\n" + "="*80)
