"""
Unit tests for models.
"""
import pytest
import torch
from src.models.vision_model import VisionEncoder, CNNEncoder
from src.models.nlp_model import SentimentClassifier
from src.models.fusion_model import MultimodalFusionModel


class TestVisionModel:
    """Test cases for vision models."""
    
    def test_vision_encoder_forward(self):
        """Test VisionEncoder forward pass."""
        model = VisionEncoder(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        
        output = model(x)
        
        assert 'logits' in output
        assert output['logits'].shape == (2, 10)
    
    def test_vision_encoder_embeddings(self):
        """Test embedding extraction."""
        model = VisionEncoder(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        
        embeddings = model.get_embeddings(x)
        
        assert embeddings.shape == (2, model.hidden_size)
    
    def test_cnn_encoder(self):
        """Test CNN encoder."""
        model = CNNEncoder(num_classes=10, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        
        output = model(x)
        
        assert output['logits'].shape == (2, 10)


class TestNLPModel:
    """Test cases for NLP models."""
    
    def test_sentiment_classifier_forward(self):
        """Test SentimentClassifier forward pass."""
        model = SentimentClassifier(num_classes=3)
        input_ids = torch.randint(0, 30000, (2, 128))
        attention_mask = torch.ones(2, 128)
        
        output = model(input_ids, attention_mask)
        
        assert 'logits' in output
        assert output['logits'].shape == (2, 3)
    
    def test_nlp_embeddings(self):
        """Test embedding extraction."""
        model = SentimentClassifier(num_classes=3)
        input_ids = torch.randint(0, 30000, (2, 128))
        attention_mask = torch.ones(2, 128)
        
        embeddings = model.get_embeddings(input_ids, attention_mask)
        
        assert embeddings.shape == (2, model.hidden_size)


class TestFusionModel:
    """Test cases for fusion model."""
    
    def test_fusion_model_forward(self):
        """Test fusion model forward pass."""
        vision_encoder = VisionEncoder(num_classes=10)
        nlp_encoder = SentimentClassifier(num_classes=3)
        
        fusion_model = MultimodalFusionModel(
            vision_encoder=vision_encoder,
            nlp_encoder=nlp_encoder,
            num_categories=10,
            num_sentiments=3
        )
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30000, (2, 128))
        attention_mask = torch.ones(2, 128)
        
        output = fusion_model(images, input_ids, attention_mask)
        
        assert 'category_logits' in output
        assert 'sentiment_logits' in output
        assert 'recommendation_score' in output
        assert output['category_logits'].shape == (2, 10)
        assert output['sentiment_logits'].shape == (2, 3)
        assert output['recommendation_score'].shape == (2, 1)
    
    def test_fusion_embeddings(self):
        """Test fusion model embedding extraction."""
        vision_encoder = VisionEncoder(num_classes=10)
        nlp_encoder = SentimentClassifier(num_classes=3)
        
        fusion_model = MultimodalFusionModel(
            vision_encoder=vision_encoder,
            nlp_encoder=nlp_encoder,
            num_categories=10,
            num_sentiments=3
        )
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30000, (2, 128))
        attention_mask = torch.ones(2, 128)
        
        output = fusion_model(images, input_ids, attention_mask, return_embeddings=True)
        
        assert 'embeddings' in output
        assert 'vision_embeddings' in output
        assert 'nlp_embeddings' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])