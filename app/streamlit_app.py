"""
Streamlit Dashboard for Multimodal Product Review Analyzer.
"""
import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Multimodal Product Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8888"


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_multimodal(image: Image.Image, review_text: str) -> Dict:
    """Make multimodal prediction."""
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Send request
    files = {'image': ('image.png', img_byte_arr, 'image/png')}
    data = {'review_text': review_text}
    
    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.text}")
        return None


def create_confidence_chart(category: str, cat_conf: float, 
                           sentiment: str, sent_conf: float):
    """Create confidence visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[cat_conf, sent_conf],
        y=['Category', 'Sentiment'],
        orientation='h',
        marker=dict(
            color=['#1f77b4', '#ff7f0e'],
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        text=[f'{cat_conf:.1%}', f'{sent_conf:.1%}'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Confidence Score",
        yaxis_title="Task",
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        showlegend=False
    )
    
    return fig


def create_recommendation_gauge(score: float):
    """Create recommendation gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Recommendation Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#ffcccc'},
                {'range': [33, 66], 'color': '#ffffcc'},
                {'range': [66, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🛍️ Multimodal Product Review Analyzer")
    st.markdown("""
    This application uses advanced AI to analyze product images and customer reviews simultaneously,
    providing comprehensive insights into product categories, customer sentiment, and purchase recommendations.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # API health check
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Offline")
        st.error("Cannot connect to API. Please ensure the FastAPI server is running on port 8000.")
        st.info("Run: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`")
        return
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Multimodal (Image + Text)", "Image Only", "Text Only"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Upload a product image
    2. Enter a customer review
    3. Click 'Analyze Product'
    4. View comprehensive results
    """)
    
    # Main content area
    if mode == "Multimodal (Image + Text)":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Product Image")
            uploaded_image = st.file_uploader(
                "Upload product image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of the product"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("📝 Customer Review")
            review_text = st.text_area(
                "Enter product review",
                height=200,
                placeholder="Type the customer review here...",
                help="Enter a detailed product review"
            )
            
            # Example reviews
            if st.button("Load Example Review"):
                review_text = st.text_area(
                    "Enter product review",
                    value="This product exceeded my expectations! The quality is outstanding and it arrived quickly. Highly recommend to anyone looking for a reliable option.",
                    height=200
                )
        
        # Analysis button
        st.markdown("---")
        if st.button("🚀 Analyze Product", type="primary", use_container_width=True):
            if not uploaded_image:
                st.warning("Please upload a product image.")
            elif not review_text or len(review_text.strip()) < 10:
                st.warning("Please enter a review (at least 10 characters).")
            else:
                with st.spinner("Analyzing product... Please wait."):
                    result = predict_multimodal(image, review_text)
                
                if result and result.get('success'):
                    st.success("✅ Analysis Complete!")
                    
                    # Results section
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Key metrics in columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label="Product Category",
                            value=result['category'],
                            delta=f"{result['category_confidence']:.1%} confidence"
                        )
                    
                    with metric_col2:
                        sentiment_emoji = {
                            'positive': '😊',
                            'neutral': '😐',
                            'negative': '😞'
                        }
                        st.metric(
                            label="Customer Sentiment",
                            value=f"{sentiment_emoji.get(result['sentiment'], '😐')} {result['sentiment'].title()}",
                            delta=f"{result['sentiment_confidence']:.1%} confidence"
                        )
                    
                    with metric_col3:
                        rec_score = result['recommendation_score']
                        if rec_score > 0.7:
                            rec_label = "High"
                            rec_color = "normal"
                        elif rec_score >= 0.45:
                            rec_label = "Medium"
                            rec_color = "normal"
                        else:
                            rec_label = "Low"
                            rec_color = "inverse"
                        
                        st.metric(
                            label="Recommendation",
                            value=rec_label,
                            delta=f"{rec_score:.1%} score"
                        )
                    
                    # Show star rating if available
                    if 'star_rating' in result and result['star_rating']:
                        st.info(f"⭐ **Star Rating**: {result['star_rating']}/5 stars")
                    
                    # Visualizations
                    viz_col1, viz_col2 = st.columns([1, 1])
                    
                    with viz_col1:
                        conf_chart = create_confidence_chart(
                            result['category'],
                            result['category_confidence'],
                            result['sentiment'],
                            result['sentiment_confidence']
                        )
                        st.plotly_chart(conf_chart, use_container_width=True)
                    
                    with viz_col2:
                        gauge_chart = create_recommendation_gauge(result['recommendation_score'])
                        st.plotly_chart(gauge_chart, use_container_width=True)
                    
                    # Show all probabilities if available
                    if 'all_sentiment_probabilities' in result and result['all_sentiment_probabilities']:
                        st.markdown("---")
                        st.subheader("📊 Sentiment Breakdown")
                        sent_col1, sent_col2, sent_col3 = st.columns(3)
                        
                        with sent_col1:
                            neg_prob = result['all_sentiment_probabilities'].get('negative', 0)
                            st.metric("😞 Negative", f"{neg_prob:.1%}")
                        
                        with sent_col2:
                            neu_prob = result['all_sentiment_probabilities'].get('neutral', 0)
                            st.metric("😐 Neutral", f"{neu_prob:.1%}")
                        
                        with sent_col3:
                            pos_prob = result['all_sentiment_probabilities'].get('positive', 0)
                            st.metric("😊 Positive", f"{pos_prob:.1%}")

                    # Aspect level insights
                    if any(k in result for k in ['positive_aspects','negative_aspects','neutral_phrases']):
                        st.markdown("---")
                        st.subheader("🧩 Aspect Highlights")
                        aspect_col1, aspect_col2, aspect_col3 = st.columns(3)
                        pos_aspects = result.get('positive_aspects') or []
                        neg_aspects = result.get('negative_aspects') or []
                        neu_phrases = result.get('neutral_phrases') or []
                        with aspect_col1:
                            st.markdown("**Positive Aspects**")
                            if pos_aspects:
                                for a in pos_aspects:
                                    st.success(f"✓ {a.strip()}")
                            else:
                                st.write("_None detected_")
                        with aspect_col2:
                            st.markdown("**Negative Aspects**")
                            if neg_aspects:
                                for a in neg_aspects:
                                    st.error(f"✗ {a.strip()}")
                            else:
                                st.write("_None detected_")
                        with aspect_col3:
                            st.markdown("**Neutral / Mixed Phrases**")
                            if neu_phrases:
                                for a in neu_phrases:
                                    st.info(f"≈ {a.strip()}")
                            else:
                                st.write("_None detected_")
                    
                    # Detailed insights
                    st.markdown("---")
                    st.subheader("💡 Detailed Insights")
                    
                    # Interpretation based on sentiment and recommendation
                    sentiment_type = result['sentiment'].lower()
                    
                    if sentiment_type == 'positive' and result['recommendation_score'] > 0.7:
                        st.success("""
                        **Strong Recommendation**: This product shows excellent alignment between 
                        visual quality and customer satisfaction. High confidence in positive sentiment 
                        indicates a reliable product choice.
                        """)
                    elif sentiment_type == 'neutral':
                        st.info("""
                        **Moderate Recommendation**: The review shows mixed or neutral feelings about 
                        this product. While not negative, customers may have some reservations. 
                        Consider reading more reviews for a complete picture.
                        """)
                    elif sentiment_type == 'positive' and result['recommendation_score'] > 0.4:
                        st.success("""
                        **Good Recommendation**: The product shows positive customer feedback with 
                        good alignment between image quality and reviews. A solid choice for consideration.
                        """)
                    elif sentiment_type == 'negative':
                        st.warning("""
                        **Low Recommendation**: Analysis indicates negative customer sentiment. 
                        The review suggests dissatisfaction with this product. Consider alternative 
                        options or investigate specific concerns mentioned in reviews.
                        """)
                    else:
                        st.info("""
                        **Mixed Results**: The product shows varied signals. Review confidence levels 
                        and additional customer feedback before making a purchase decision.
                        """)
    
    elif mode == "Image Only":
        st.subheader("📸 Image Analysis")
        uploaded_image = st.file_uploader("Upload product image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        
                        files = {'image': ('image.png', img_byte_arr, 'image/png')}
                        response = requests.post(f"{API_URL}/predict/vision", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"**Category**: {result['category']}")
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                            
                            # Show all probabilities
                            st.subheader("All Category Probabilities")
                            probs_df = {k: f"{v:.2%}" for k, v in result['all_probabilities'].items()}
                            st.json(probs_df)
    
    else:  # Text Only
        st.subheader("📝 Sentiment Analysis")
        review_text = st.text_area("Enter product review", height=200)
        
        if st.button("Analyze Sentiment", type="primary"):
            if review_text and len(review_text.strip()) >= 10:
                with st.spinner("Analyzing sentiment..."):
                    data = {'review_text': review_text}
                    response = requests.post(f"{API_URL}/predict/nlp", data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        sentiment_emoji = {
                            'positive': '😊',
                            'neutral': '😐',
                            'negative': '😞'
                        }
                        
                        st.success(f"**Sentiment**: {sentiment_emoji.get(result['sentiment'], '😐')} {result['sentiment'].title()}")
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        # Show all probabilities
                        st.subheader("Sentiment Breakdown")
                        probs_df = {k.title(): f"{v:.2%}" for k, v in result['all_probabilities'].items()}
                        st.json(probs_df)
            else:
                st.warning("Please enter a review (at least 10 characters).")


if __name__ == "__main__":
    main()