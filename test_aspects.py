"""Test aspect extraction fix."""
import sys
sys.path.insert(0, 'c:/Users/User/Documents/multimodal-product-analyzer')

from src.models.pretrained_fusion import PretrainedFusionModel
from transformers import pipeline, CLIPProcessor, CLIPModel

print("Loading models...")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Books', 'Toys', 'Beauty', 'Automotive', 'Food', 'Health']
sentiments = ['negative', 'neutral', 'positive']
fusion_model = PretrainedFusionModel(clip_model, clip_processor, sentiment_pipeline, categories, sentiments)

review = """I recently used the Oppo Reno 12 Pro 5G, and overall it's a pretty balanced device. On the plus side, it has a 120 Hz AMOLED display that's smooth and crisp, making everyday use and media consumption very pleasant. The battery life is solid, and the 80 W fast charging helps a lot - a short charge gives a good amount of juice."""

print("\n" + "="*80)
print("TESTING ASPECT EXTRACTION")
print("="*80)

analysis = fusion_model._analyze_sentiment_comprehensive(review)

print(f"\nSentiment: {analysis['final_star_rating']} stars")
print(f"Rule: {analysis.get('rule_triggered', 'N/A')}")

print(f"\n✅ POSITIVE ASPECTS ({len(analysis['positive_aspects'])}):")
for aspect in analysis['positive_aspects']:
    print(f"   + {aspect}")

print(f"\n❌ NEGATIVE ASPECTS ({len(analysis['negative_aspects'])}):")
for aspect in analysis['negative_aspects']:
    print(f"   - {aspect}")

print(f"\n⚖️  NEUTRAL/MIXED PHRASES ({len(analysis['neutral_phrases'])}):")
for phrase in analysis['neutral_phrases']:
    print(f"   ~ {phrase}")

print("\n" + "="*80)
print("Expected:")
print("  POSITIVE: Display sentence (smooth, crisp, pleasant)")
print("  POSITIVE: Battery sentence (solid, helps)")
print("  NEUTRAL: Opening sentence (balanced device, overall)")
print("="*80)
