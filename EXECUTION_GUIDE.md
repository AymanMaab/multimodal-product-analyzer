# Complete Execution Guide - Multimodal Product Review Analyzer

This guide provides step-by-step instructions to execute the complete project from scratch.

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- 16GB RAM (minimum)
- 50GB free disk space
- NVIDIA GPU with 8GB+ VRAM (recommended for training)
- Git installed
- Internet connection for downloading dependencies

## 🚀 Method 1: Automated Setup (Recommended)

### Step 1: Clone and Setup

```bash
# Clone the repository (or create project directory)
mkdir multimodal-product-analyzer
cd multimodal-product-analyzer

# Copy all provided files into this directory
# Ensure the structure matches the PROJECT_SUMMARY.md

# Make setup script executable
chmod +x quickstart.sh

# Run automated setup
./quickstart.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download NLTK data
- ✅ Create directory structure
- ✅ Generate sample dataset (1000 samples)
- ✅ Preprocess data

### Step 2: Train Models (Quick Test with Small Dataset)

```bash
# Activate virtual environment
source venv/bin/activate

# Train vision model (5 epochs for quick testing)
python scripts/train_vision.py \
    --data_csv data/processed/dataset.csv \
    --epochs 5 \
    --batch_size 16 \
    --device cuda

# Train NLP model (3 epochs for quick testing)
python scripts/train_nlp.py \
    --data_csv data/processed/dataset.csv \
    --epochs 3 \
    --batch_size 8 \
    --device cuda

# Train fusion model (5 epochs for quick testing)
python scripts/train_fusion.py \
    --data_csv data/processed/dataset.csv \
    --vision_checkpoint models/vision/best_acc.pth \
    --nlp_checkpoint models/nlp/best_acc.pth \
    --epochs 5 \
    --batch_size 16 \
    --freeze_encoders \
    --device cuda
```

**Expected Time**: 
- Vision: ~10 minutes
- NLP: ~5 minutes  
- Fusion: ~8 minutes
- **Total: ~25 minutes**

### Step 3: Launch Application

**Terminal 1 - API Server:**
```bash
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Streamlit Dashboard:**
```bash
source venv/bin/activate
streamlit run app/streamlit_app.py
```

### Step 4: Test the System

Open browser:
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

Test prediction:
1. Upload a product image from `data/raw/images/`
2. Enter a review text
3. Click "Analyze Product"
4. View results with confidence scores and visualization

---

## 🔧 Method 2: Manual Step-by-Step

### Phase 1: Environment Setup

```bash
# Create project directory
mkdir multimodal-product-analyzer
cd multimodal-product-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Phase 2: Data Preparation

```bash
# Create directories
mkdir -p data/raw data/processed models/{vision,nlp,fusion} logs results/figures

# Generate sample dataset
python scripts/download_data.py --create_sample --output data/raw

# Verify dataset
ls data/raw/images/ | wc -l  # Should show 1000
head data/raw/reviews.csv

# Preprocess data
python scripts/preprocess_data.py \
    --images_dir data/raw/images \
    --reviews_csv data/raw/reviews.csv \
    --output data/processed/dataset.csv

# Check preprocessed data
head data/processed/dataset.csv
```

### Phase 3: Model Training

#### 3a. Train Vision Model (Week 2)

```bash
python scripts/train_vision.py \
    --data_csv data/processed/dataset.csv \
    --model_type vit \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda

# Monitor training
tensorboard --logdir models/vision/tensorboard
```

**Expected Output:**
```
Epoch 30/30 - Train Loss: 0.243, Train Acc: 0.921, Val Loss: 0.312, Val Acc: 0.898
Training completed!
Best validation accuracy: 0.898
```

#### 3b. Train NLP Model (Week 2)

```bash
python scripts/train_nlp.py \
    --data_csv data/processed/dataset.csv \
    --model_type bert \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5 \
    --device cuda
```

**Expected Output:**
```
Epoch 10/10 - Train Loss: 0.325, Train Acc: 0.873, Val Loss: 0.389, Val Acc: 0.851
Training completed!
Best validation F1-score: 0.849
```

#### 3c. Train Fusion Model (Week 3)

```bash
python scripts/train_fusion.py \
    --data_csv data/processed/dataset.csv \
    --vision_checkpoint models/vision/best_acc.pth \
    --nlp_checkpoint models/nlp/best_acc.pth \
    --epochs 20 \
    --batch_size 32 \
    --freeze_encoders \
    --device cuda
```

**Expected Output:**
```
Epoch 20/20 - Train Loss: 0.287, Val Loss: 0.341
Category Acc: 0.901, Sentiment F1: 0.858, Rec MAE: 0.095
Training completed!
```

### Phase 4: Model Evaluation (Week 3)

```bash
python scripts/evaluate.py \
    --data_csv data/processed/dataset.csv \
    --fusion_checkpoint models/fusion/best_acc.pth \
    --batch_size 32 \
    --device cuda \
    --output results

# View results
cat results/evaluation_results.json
ls results/figures/  # Confusion matrices and plots
```

### Phase 5: Deployment (Week 4)

#### 5a. Start API Server

```bash
# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Development mode (with auto-reload)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test API:
```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -F "image=@data/raw/images/00001.jpg" \
  -F "review_text=Excellent product, highly recommend!"
```

#### 5b. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Access at: http://localhost:8501

---

## 🐳 Method 3: Docker Deployment (Production)

### Step 1: Build Images

```bash
cd docker
docker-compose build
```

### Step 2: Start Services

```bash
docker-compose up -d
```

### Step 3: Verify Services

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs -f

# Test API
curl http://localhost:8000/health

# Access dashboard
open http://localhost:8501
```

### Step 4: Stop Services

```bash
docker-compose down
```

---

## 📊 Verification Checklist

After execution, verify:

- [ ] **Data Generated**: `data/raw/` contains 1000 images and CSV
- [ ] **Data Preprocessed**: `data/processed/dataset.csv` exists
- [ ] **Models Trained**: 
  - [ ] `models/vision/best_acc.pth` exists
  - [ ] `models/nlp/best_acc.pth` exists
  - [ ] `models/fusion/best_acc.pth` exists
- [ ] **Logs Created**: `models/*/tensorboard/` contains training logs
- [ ] **API Running**: `curl http://localhost:8000/health` returns 200
- [ ] **Dashboard Accessible**: Can open http://localhost:8501
- [ ]