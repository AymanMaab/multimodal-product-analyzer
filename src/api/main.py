"""
FastAPI backend for Multimodal Product Review Analyzer.
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import torch
from PIL import Image
import io
import numpy as np
from transformers import AutoTokenizer, pipeline, CLIPProcessor, CLIPModel

from src.models.pretrained_fusion import PretrainedFusionModel
from src.config import config

# Initialize FastAPI app
app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    description=config.api.description
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
fusion_model = None
sentiment_pipeline = None  # Pre-trained sentiment analyzer
clip_model = None  # Pre-trained vision classifier
clip_processor = None

# Label mappings
CATEGORIES = config.data.categories
SENTIMENTS = config.data.sentiment_labels


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    category: str
    category_confidence: float
    sentiment: str
    sentiment_confidence: float
    recommendation_score: float
    success: bool
    message: Optional[str] = None
    all_category_probabilities: Optional[Dict[str, float]] = None
    all_sentiment_probabilities: Optional[Dict[str, float]] = None
    star_rating: Optional[int] = None
    rule_triggered: Optional[str] = None
    positive_aspects: Optional[list[str]] = None
    negative_aspects: Optional[list[str]] = None
    neutral_phrases: Optional[list[str]] = None
    base_star_avg: Optional[float] = None
    base_chunks: Optional[int] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: bool
    version: str


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global vision_model, nlp_model, fusion_model, tokenizer, image_transform, sentiment_pipeline, clip_model, clip_processor
    
    try:
        print("Loading models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained sentiment analyzer for better predictions
        print("Loading sentiment analysis pipeline...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load CLIP for zero-shot image classification
        print("Loading CLIP vision model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create pretrained fusion model wrapper
        print("Creating pretrained fusion model...")
        fusion_model = PretrainedFusionModel(
            clip_model=clip_model,
            clip_processor=clip_processor,
            sentiment_pipeline=sentiment_pipeline,
            categories=CATEGORIES,
            sentiments=SENTIMENTS
        ).to(device)
        fusion_model.eval()
        
        print("All models loaded successfully!")
        print("✓ Using pretrained fusion model (CLIP + Sentiment analyzer)")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def classify_image_with_clip(pil_image: Image.Image) -> tuple:
    """
    Classify image using CLIP zero-shot classification.
    Returns (category, confidence)
    """
    # Create detailed text prompts for each category
    category_prompts = {
        "Electronics": "a photo of electronic device like smartphone, laptop, tablet, camera, headphones, or computer",
        "Clothing": "a photo of clothing item like shirt, pants, dress, shoes, or accessories",
        "Home & Kitchen": "a photo of home appliance, furniture, kitchenware, or home decor",
        "Sports": "a photo of sports equipment, fitness gear, or athletic item",
        "Books": "a photo of a book, magazine, or printed reading material",
        "Toys": "a photo of children's toy, doll, action figure, or game",
        "Beauty": "a photo of cosmetics, makeup, skincare product, or beauty item",
        "Automotive": "a photo of car parts, automotive accessories, or vehicle equipment",
        "Food": "a photo of packaged food, snacks, or beverages",
        "Health": "a photo of medical device, supplements, or health product"
    }
    
    # Get prompts in correct order
    prompts = [category_prompts[cat] for cat in CATEGORIES]
    
    print(f"[CLIP] Classifying image with {len(prompts)} categories")
    
    # Process image and text
    inputs = clip_processor(
        text=prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True
    ).to(next(clip_model.parameters()).device)
    
    # Get predictions
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
    
    # Print all probabilities for debugging
    print("[CLIP] Category probabilities:")
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat}: {probs[i].item():.4f}")
    
    # Get best prediction
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    
    print(f"[CLIP] Final prediction: {CATEGORIES[pred_idx]} (confidence: {confidence:.4f})")
    
    return CATEGORIES[pred_idx], confidence


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "online",
        "models_loaded": fusion_model is not None,
        "version": config.api.version
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if fusion_model is not None else "unhealthy",
        "models_loaded": fusion_model is not None,
        "version": config.api.version
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    review_text: str = Form(...)
):
    """
    Predict product category, sentiment, and recommendation score.
    
    Args:
        image: Product image file
        review_text: Customer review text
    
    Returns:
        Prediction results
    """
    try:
        if fusion_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Use fusion model for predictions
        with torch.no_grad():
            outputs = fusion_model([pil_image], [review_text])
        
        # Extract category prediction
        category_logits = outputs['category_logits'][0]
        category_probs = torch.softmax(category_logits, dim=0)
        category_idx = torch.argmax(category_probs).item()
        category = CATEGORIES[category_idx]
        category_conf = category_probs[category_idx].item()
        
        # Use comprehensive sentiment analysis with aspect extraction
        analysis = fusion_model._analyze_sentiment_comprehensive(review_text)
        final_star = analysis['final_star_rating']
        if final_star >= 4:
            sentiment = 'positive'
        elif final_star == 3:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'
        sentiment_conf = float(analysis['confidence'])
        # Build probability distribution from analysis scores
        pos_score = max(analysis.get('positive_score', 0.0), 0.0)
        neg_score = max(analysis.get('negative_score', 0.0), 0.0)
        neu_score = max(analysis.get('neutral_score', 0.0), 0.0)
        total_score = pos_score + neg_score + neu_score
        if total_score <= 0:
            total_score = 1.0
        all_sentiment_probs = {
            'negative': neg_score / total_score,
            'neutral': neu_score / total_score,
            'positive': pos_score / total_score
        }
        
        # Extract recommendation score
        recommendation_score = outputs['recommendation_score'][0].item()
        
        # Star rating derived from comprehensive analysis
        star_rating = final_star
        
        # Build probability dictionaries
        all_category_probs = {CATEGORIES[i]: float(category_probs[i]) for i in range(len(CATEGORIES))}
        # all_sentiment_probabilities replaced by comprehensive analysis distribution
        all_sentiment_probs = all_sentiment_probs
        
        return PredictionResponse(
            category=category,
            category_confidence=float(category_conf),
            sentiment=sentiment,
            sentiment_confidence=float(sentiment_conf),
            recommendation_score=float(recommendation_score),
            all_category_probabilities=all_category_probs,
            all_sentiment_probabilities=all_sentiment_probs,
            star_rating=star_rating,
            rule_triggered=analysis.get('rule_triggered'),
            positive_aspects=analysis.get('positive_aspects'),
            negative_aspects=analysis.get('negative_aspects'),
            neutral_phrases=analysis.get('neutral_phrases'),
            base_star_avg=analysis.get('base_star_avg'),
            base_chunks=analysis.get('base_chunks'),
            success=True,
            message="Prediction successful"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/vision")
async def predict_vision_only(image: UploadFile = File(...)):
    """Predict product category from image only."""
    try:
        if clip_model is None:
            raise HTTPException(status_code=503, detail="Vision model not loaded")
        
        # Process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Use CLIP for classification
        category, confidence = classify_image_with_clip(pil_image)
        
        # Get all probabilities using the same detailed prompts
        category_prompts = {
            "Electronics": "a photo of electronic device like smartphone, laptop, tablet, camera, headphones, or computer",
            "Clothing": "a photo of clothing item like shirt, pants, dress, shoes, or accessories",
            "Home & Kitchen": "a photo of home appliance, furniture, kitchenware, or home decor",
            "Sports": "a photo of sports equipment, fitness gear, or athletic item",
            "Books": "a photo of a book, magazine, or printed reading material",
            "Toys": "a photo of children's toy, doll, action figure, or game",
            "Beauty": "a photo of cosmetics, makeup, skincare product, or beauty item",
            "Automotive": "a photo of car parts, automotive accessories, or vehicle equipment",
            "Food": "a photo of packaged food, snacks, or beverages",
            "Health": "a photo of medical device, supplements, or health product"
        }
        prompts = [category_prompts[cat] for cat in CATEGORIES]
        
        inputs = clip_processor(
            text=prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(next(clip_model.parameters()).device)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        return {
            "category": category,
            "confidence": float(confidence),
            "all_probabilities": {
                CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision prediction error: {str(e)}")


@app.post("/predict/nlp")
async def predict_nlp_only(review_text: str = Form(...)):
    """Predict sentiment from review text only."""
    try:
        if fusion_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        # Comprehensive analysis with aspects & multi-chunk support
        analysis = fusion_model._analyze_sentiment_comprehensive(review_text)
        star_rating = analysis['final_star_rating']
        if star_rating >= 4:
            sentiment = 'positive'
        elif star_rating == 3:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'

        # Build probability distribution from analysis scores
        pos_score = max(analysis.get('positive_score', 0.0), 0.0)
        neg_score = max(analysis.get('negative_score', 0.0), 0.0)
        neu_score = max(analysis.get('neutral_score', 0.0), 0.0)
        total_score = pos_score + neg_score + neu_score
        if total_score <= 0:
            total_score = 1.0
        all_probs = {
            'negative': neg_score / total_score,
            'neutral': neu_score / total_score,
            'positive': pos_score / total_score
        }

        return {
            'sentiment': sentiment,
            'confidence': float(analysis['confidence']),
            'star_rating': star_rating,
            'rule_triggered': analysis.get('rule_triggered'),
            'all_probabilities': all_probs,
            'positive_aspects': analysis.get('positive_aspects'),
            'negative_aspects': analysis.get('negative_aspects'),
            'neutral_phrases': analysis.get('neutral_phrases'),
            'base_star_avg': analysis.get('base_star_avg'),
            'base_chunks': analysis.get('base_chunks')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP prediction error: {str(e)}")


@app.get("/categories")
async def get_categories():
    """Get list of supported product categories."""
    return {"categories": CATEGORIES}


@app.get("/sentiments")
async def get_sentiments():
    """Get list of sentiment labels."""
    return {"sentiments": SENTIMENTS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers
    )