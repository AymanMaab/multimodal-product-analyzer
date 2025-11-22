"""
Script to download and prepare sample dataset.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import requests
from tqdm import tqdm
import zipfile
import io
from src.config import config


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"Saved to {output_path}")


def create_sample_dataset(output_dir: Path):
    """
    Create a sample dataset for testing.
    This generates synthetic data for demonstration purposes.
    """
    import pandas as pd
    import numpy as np
    from PIL import Image
    
    print("Creating sample dataset...")
    
    # Create directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample categories and sentiments
    categories = config.data.categories
    sentiments = ['negative', 'neutral', 'positive']
    
    # Generate sample reviews
    sample_reviews = {
        'positive': [
            "This product is absolutely amazing! Exceeded all my expectations.",
            "Best purchase I've made this year. Highly recommend!",
            "Excellent quality and fast shipping. Very satisfied!",
            "Love it! Works perfectly and looks great.",
            "Outstanding product. Worth every penny!"
        ],
        'neutral': [
            "It's okay, does what it's supposed to do.",
            "Average product. Nothing special but works fine.",
            "Decent quality for the price. No complaints.",
            "It works as described. Pretty standard.",
            "Not bad, not great. Just okay."
        ],
        'negative': [
            "Very disappointed. Quality is poor.",
            "Not as described. Would not recommend.",
            "Waste of money. Broke after a week.",
            "Terrible product. Don't buy this.",
            "Not happy with this purchase at all."
        ]
    }
    
    # Generate synthetic data
    num_samples = 1000
    data = []
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Random category and sentiment
        category = np.random.choice(categories)
        sentiment = np.random.choice(sentiments)
        rating = {'negative': np.random.uniform(1, 2.5),
                 'neutral': np.random.uniform(2.5, 3.5),
                 'positive': np.random.uniform(3.5, 5)}[sentiment]
        
        # Random review
        review = np.random.choice(sample_reviews[sentiment])
        
        # Generate synthetic image (random colored square)
        img_size = 224
        img = Image.new('RGB', (img_size, img_size), 
                       color=tuple(np.random.randint(0, 255, 3).tolist()))
        img_path = images_dir / f"{i:05d}.jpg"
        img.save(img_path)
        
        # Compute recommendation score
        rec_score = (rating - 1) / 4.0 + np.random.uniform(-0.1, 0.1)
        rec_score = np.clip(rec_score, 0, 1)
        
        data.append({
            'product_id': i,
            'image_path': str(img_path),
            'review_text': review,
            'category': category,
            'sentiment': sentiment,
            'rating': rating,
            'recommendation_score': rec_score,
            'helpful_votes': np.random.randint(0, 50),
            'total_votes': np.random.randint(0, 100)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    csv_path = output_dir / "reviews.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nSample dataset created:")
    print(f"  Images: {images_dir}")
    print(f"  Reviews CSV: {csv_path}")
    print(f"  Total samples: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())


def parse_args():
    parser = argparse.ArgumentParser(description="Download or create sample dataset")
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--image_url', type=str, default=None,
                       help='URL to download images (optional)')
    parser.add_argument('--review_url', type=str, default=None,
                       help='URL to download reviews CSV (optional)')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create synthetic sample dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.create_sample:
        # Create synthetic sample dataset
        create_sample_dataset(output_dir)
    else:
        # Download from URLs
        if args.image_url:
            image_zip = output_dir / "images.zip"
            download_file(args.image_url, image_zip)
            
            # Extract
            print("Extracting images...")
            with zipfile.ZipFile(image_zip, 'r') as zip_ref:
                zip_ref.extractall(output_dir / "images")
            print("Images extracted!")
        
        if args.review_url:
            reviews_csv = output_dir / "reviews.csv"
            download_file(args.review_url, reviews_csv)
        
        if not args.image_url and not args.review_url:
            print("No URLs provided. Use --create_sample to generate synthetic data.")
            print("\nExample usage:")
            print("  python scripts/download_data.py --create_sample")
            print("  python scripts/download_data.py --image_url URL --review_url URL")


if __name__ == "__main__":
    main()