"""
Configuration management for Multimodal Product Review Analyzer.
"""
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path configurations."""
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # Model paths
    MODELS_DIR: Path = BASE_DIR / "models"
    VISION_MODEL_PATH: Path = MODELS_DIR / "vision_model.pth"
    NLP_MODEL_PATH: Path = MODELS_DIR / "nlp_model.pth"
    FUSION_MODEL_PATH: Path = MODELS_DIR / "fusion_model.pth"
    
    # Log paths
    LOGS_DIR: Path = BASE_DIR / "logs"
    TENSORBOARD_DIR: Path = LOGS_DIR / "tensorboard"
    
    # Results paths
    RESULTS_DIR: Path = BASE_DIR / "results"
    FIGURES_DIR: Path = RESULTS_DIR / "figures"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [
            self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
            self.MODELS_DIR, self.LOGS_DIR, self.TENSORBOARD_DIR,
            self.RESULTS_DIR, self.FIGURES_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class VisionConfig:
    """Vision model configurations."""
    # Model architecture
    model_name: str = "google/vit-base-patch16-224"  # Vision Transformer
    num_classes: int = 10  # Product categories
    img_size: int = 224
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # Preprocessing
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class NLPConfig:
    """NLP model configurations."""
    # Model architecture
    model_name: str = "bert-base-uncased"
    num_classes: int = 3  # Positive, Neutral, Negative
    max_length: int = 128
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    
    # Text preprocessing
    lowercase: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = True
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class FusionConfig:
    """Fusion model configurations."""
    # Architecture
    vision_embed_dim: int = 768  # ViT output dimension
    nlp_embed_dim: int = 768  # BERT output dimension
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Loss weights (multi-task learning)
    category_weight: float = 1.0
    sentiment_weight: float = 1.0
    recommendation_weight: float = 2.0
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class DataConfig:
    """Data configurations."""
    # Dataset split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Dataset URLs (placeholders - replace with actual URLs)
    image_dataset_url: str = "https://example.com/product_images.zip"
    review_dataset_url: str = "https://example.com/product_reviews.csv"
    
    # Data sampling
    random_seed: int = 42
    sample_size: Optional[int] = None  # None for full dataset
    
    # Product categories
    categories: List[str] = field(default_factory=lambda: [
        "Electronics", "Clothing", "Home & Kitchen", "Sports",
        "Books", "Toys", "Beauty", "Automotive", "Food", "Health"
    ])
    
    # Sentiment labels
    sentiment_labels: List[str] = field(default_factory=lambda: [
        "negative", "neutral", "positive"
    ])


@dataclass
class APIConfig:
    """API configurations."""
    title: str = "Multimodal Product Review Analyzer API"
    version: str = "1.0.0"
    description: str = "Production-ready API for multimodal product analysis"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    
    # File upload
    max_upload_size_mb: int = 10


@dataclass
class StreamlitConfig:
    """Streamlit app configurations."""
    page_title: str = "Multimodal Product Analyzer"
    page_icon: str = "🛍️"
    layout: str = "wide"
    
    # API endpoint
    api_url: str = "http://localhost:8000"
    
    # UI settings
    theme: str = "dark"
    show_progress: bool = True


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.vision = VisionConfig()
        self.nlp = NLPConfig()
        self.fusion = FusionConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.streamlit = StreamlitConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        # Update configurations from YAML
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


# Global configuration instance
config = Config()