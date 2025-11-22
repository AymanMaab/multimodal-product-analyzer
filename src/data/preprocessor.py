"""
Data preprocessing utilities for text and images.
"""
import re
import string
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """
    Text preprocessing for product reviews.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = True
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
        """
        Process text column in DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
        
        Returns:
            DataFrame with cleaned text
        """
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # Remove empty reviews
        df = df[df[text_column].str.len() > 0]
        
        return df


class ImagePreprocessor:
    """
    Image preprocessing utilities.
    """
    
    def __init__(
        self,
        target_size: tuple = (224, 224),
        ensure_rgb: bool = True
    ):
        self.target_size = target_size
        self.ensure_rgb = ensure_rgb
    
    def process_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Processed PIL Image or None if error
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if self.ensure_rgb and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(self.target_size, Image.LANCZOS)
            
            return image
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def validate_images(self, image_paths: List[Path]) -> List[bool]:
        """
        Validate a list of image paths.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of boolean values indicating validity
        """
        valid = []
        for path in image_paths:
            try:
                img = Image.open(path)
                img.verify()  # Verify image integrity
                valid.append(True)
            except Exception:
                valid.append(False)
        
        return valid


def analyze_sentiment_from_rating(rating: float) -> str:
    """
    Convert numerical rating to sentiment label.
    
    Args:
        rating: Rating value (e.g., 1-5 stars)
    
    Returns:
        Sentiment label: 'positive', 'neutral', or 'negative'
    """
    if rating >= 4.0:
        return 'positive'
    elif rating >= 3.0:
        return 'neutral'
    else:
        return 'negative'


def compute_recommendation_score(
    rating: float,
    helpful_votes: int = 0,
    total_votes: int = 1
) -> float:
    """
    Compute recommendation score from rating and helpfulness.
    
    Args:
        rating: Product rating (1-5)
        helpful_votes: Number of helpful votes
        total_votes: Total number of votes
    
    Returns:
        Recommendation score between 0 and 1
    """
    # Normalize rating to 0-1
    rating_score = (rating - 1) / 4.0
    
    # Compute helpfulness ratio
    if total_votes > 0:
        helpfulness_score = helpful_votes / total_votes
    else:
        helpfulness_score = 0.5
    
    # Weighted combination
    recommendation = 0.7 * rating_score + 0.3 * helpfulness_score
    
    return np.clip(recommendation, 0.0, 1.0)


def prepare_dataset(
    images_dir: Path,
    reviews_csv: Path,
    output_csv: Path,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Prepare complete dataset by linking images with reviews.
    
    Args:
        images_dir: Directory containing product images
        reviews_csv: Path to reviews CSV file
        output_csv: Path to save processed dataset
        sample_size: Optional sample size for testing
    
    Returns:
        Processed DataFrame
    """
    # Load reviews
    df = pd.read_csv(reviews_csv)
    
    # Sample if needed
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Initialize preprocessors
    text_processor = TextPreprocessor()
    image_processor = ImagePreprocessor()
    
    # Process text
    print("Processing text reviews...")
    df = text_processor.process_dataframe(df)
    
    # Link images (assuming image filenames match product IDs)
    print("Linking images...")
    image_paths = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        # Try multiple image extensions
        product_id = row.get('product_id', idx)
        found = False
        
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            img_path = images_dir / f"{product_id}{ext}"
            if img_path.exists():
                image_paths.append(str(img_path))
                valid_indices.append(idx)
                found = True
                break
        
        if not found:
            # Use a placeholder or skip
            continue
    
    # Filter dataframe to only valid image entries
    df = df.loc[valid_indices].copy()
    df['image_path'] = image_paths
    
    # Add sentiment labels if not present
    if 'sentiment' not in df.columns and 'rating' in df.columns:
        df['sentiment'] = df['rating'].apply(analyze_sentiment_from_rating)
    
    # Add recommendation score if not present
    if 'recommendation_score' not in df.columns and 'rating' in df.columns:
        df['recommendation_score'] = df.apply(
            lambda row: compute_recommendation_score(
                row['rating'],
                row.get('helpful_votes', 0),
                row.get('total_votes', 1)
            ),
            axis=1
        )
    
    # Save processed dataset
    df.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to {output_csv}")
    print(f"Total samples: {len(df)}")
    
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df['category'] if 'category' in df.columns else None
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df['category'] if 'category' in train_val_df.columns else None
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df