import streamlit as st
import joblib
import re
import string
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import base64

# -------------------------------
# Custom CSS Styling
# -------------------------------
def local_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
            html, body, [class*="css"]  {
                font-family: 'Poppins', sans-serif !important;
                background: #fff !important;
                min-height: 100vh;
            }
            .main {
                background: #fff !important;
            }
            .container {
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
                padding-left: 0;
                padding-right: 0;
            }
            .navbar {
                width: 100%;
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 18px 0 12px 0;
                background: #fff;
                box-shadow: none;
                border-bottom: 1.5px solid #D7E9F7;
            }
            .navbar-left {
                display: flex;
                align-items: center;
                gap: 14px;
            }
            .navbar-logo img {
                width: 38px;
                height: 38px;
                border-radius: 10px;
            }
            .navbar-title {
                font-size: 1.35rem;
                font-weight: 700;
                color: #062659;
                font-family: 'Poppins', sans-serif !important;
            }
            .navbar-right {
                display: flex;
                align-items: center;
                gap: 32px;
            }
            .navbar-links {
                display: flex;
                gap: 28px;
                font-size: 17px;
                font-weight: 500;
            }
            .navbar-links a {
                color: #062659;
                text-decoration: none;
                transition: color 0.2s;
                font-family: 'Poppins', sans-serif !important;
            }
            .navbar-links a:hover {
                color: #F6564D;
            }
            .signup-btn {
                background: #fff;
                color: #F6564D;
                border: 2px solid #F6564D;
                border-radius: 8px;
                padding: 8px 22px;
                font-size: 17px;
                font-family: 'Poppins', sans-serif !important;
                font-weight: 600;
                box-shadow: 0 2px 8px rgba(246,86,77,0.08);
                transition: background 0.2s, color 0.2s, border 0.2s;
                cursor: pointer;
            }
            .signup-btn:hover {
                background: #F6564D;
                color: #fff;
                border: 2px solid #F6564D;
            }
            .hero {
                display: flex;
                flex-wrap: wrap;
                align-items: flex-start;
                justify-content: space-between;
                margin-top: 24px;
                margin-bottom: 32px;
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
            }
            .hero-content {
                max-width: 600px;
                text-align: left;
            }
            .hero-title {
                font-size: 3.2rem;
                font-weight: 700;
                color: #062659;
                margin-bottom: 0.5rem;
                line-height: 1.1;
                font-family: 'Poppins', sans-serif !important;
            }
            .hero-highlight {
                color: #F6564D;
                font-weight: 700;
                display: inline-block;
                border-bottom: 5px solid #F6564D;
                margin-bottom: 0.5rem;
                font-family: 'Poppins', sans-serif !important;
            }
            .hero-subtitle {
                font-size: 1.25rem;
                color: #062659;
                margin-bottom: 2rem;
                font-family: 'Poppins', sans-serif !important;
            }
            .stTextArea textarea {
                border-radius: 12px;
                font-size: 16px;
                font-family: 'Poppins', sans-serif !important;
                background: #D7E9F7;
                border: 2px solid #062659;
                padding: 18px;
                color: #062659;
                box-shadow: 0 2px 12px rgba(6,38,89,0.08);
                width: 100% !important;
                min-width: 0;
                max-width: 1200px;
                margin-bottom: 0.5rem;
            }
            .stTextArea textarea::placeholder {
                color: #94a3b8;
                font-size: 16px;
                font-family: 'Poppins', sans-serif !important;
            }
            .stButton > button {
                background: #F6564D;
                color: white;
                border-radius: 8px;
                padding: 14px 28px;
                font-size: 18px;
                font-family: 'Poppins', sans-serif !important;
                font-weight: 600;
                box-shadow: 0 2px 8px rgba(246,86,77,0.10);
                border: none;
                transition: background 0.2s, box-shadow 0.2s;
            }
            .stButton > button:hover {
                background: #062659;
                color: #fff;
                box-shadow: 0 4px 16px rgba(6,38,89,0.12);
            }
            .stSelectbox > div {
                font-family: 'Poppins', sans-serif !important;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
                font-family: 'Poppins', sans-serif !important;
                font-weight: 700;
            }
            .stAlert {
                border-radius: 8px;
            }
            /* Make the selectbox label bigger and bolder */
            label[for^=".*choose-model.*"], .choose-model-label {
                font-size: 1.35rem !important;
                font-weight: 700 !important;
                color: #062659 !important;
                margin-bottom: 0.25rem !important;
                font-family: 'Poppins', sans-serif !important;
                display: block;
            }
            /* Reduce the width of the selectbox */
            .stSelectbox {
                width: 300px !important;
                min-width: 0;
                max-width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

local_css()

# -------------------------------
# Clean Text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -------------------------------
# Load Vectorizer
# -------------------------------
try:
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error(f"❌ Failed to load vectorizer: {e}")
    st.stop()

# -------------------------------
# Load Model Based on Selection
# -------------------------------
def load_model(model_name):
    try:
        if model_name == "XGBoost":
            return joblib.load("xgb_fraud_detector.pkl")
        elif model_name == "Random Forest":
            return joblib.load("job_post_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# -------------------------------
# Predict Function
# -------------------------------
def predict_fraudulent(job_post_text, model):
    cleaned_text = clean_text(job_post_text)
    vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized)
    return prediction[0], vectorized

# -------------------------------
# SHAP Explainability
# -------------------------------
def get_explainer(model):
    return shap.TreeExplainer(model)


def explain_prediction(text, model, explainer):
    cleaned_text = clean_text(text)
    vectorized = vectorizer.transform([cleaned_text])
    shap_values = explainer.shap_values(vectorized.toarray(), check_additivity=False)
    feature_names = vectorizer.get_feature_names_out()

    if isinstance(shap_values, list):
        scores = shap_values[1][0]  # For binary classifier (fraud class)
    else:
        scores = shap_values[0]


    top_indices = np.argsort(np.abs(scores))[::-1][:10]
    top_features = []
    for i in top_indices:
        value = scores[i]
        # If value is a numpy array, flatten and get the first element
        if isinstance(value, np.ndarray):
            value = value.flatten()[0]
        value = float(value)
        top_features.append((feature_names[i], value))
    return top_features



# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake Job Post Detector", layout="wide")
local_css()

# Navigation bar
st.markdown("""
<div class='container'>
  <div class="navbar">
      <div class="navbar-left">
          <span class="navbar-logo"><img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="Logo"></span>
          <span class="navbar-title">Fake Job Detector</span>
      </div>
      <div class="navbar-right">
          <div class="navbar-links">
              <a href="#">Home</a>
              <a href="#">Contact Us</a>
              <a href="#">About Us</a>
              <a href="#">Sign In</a>
          </div>
          <button class="signup-btn">Sign Up</button>
      </div>
  </div>

  <div class="hero">
      <div class="hero-content">
          <div class="hero-title">Fake Job Detector<br><span class="hero-highlight">stop wasting your time!</span></div>
          <div class="hero-subtitle">Spot fake job offers instantly with AI-powered accuracy..</div>
      </div>
      <div>
          <img src=\"https://fastcompanyme.com/wp-content/uploads/2023/03/Why-are-employment-scams-on-the-rise-in-MEA.jpeg\" alt=\"Fake Job Illustration\" style=\"max-width: 480px; margin-top: 32px; border-radius: 18px; box-shadow: 0 4px 24px rgba(6,38,89,0.10);\">
      </div>
  </div>
""", unsafe_allow_html=True)

# For the form and main content, do not wrap in a separate container, just continue in the same container

# Left-aligned form fields
st.markdown('<span class="choose-model-label">Choose Model</span>', unsafe_allow_html=True)
model_choice = st.selectbox("", ["XGBoost", "Random Forest"], key="choose-model")
if st.button("🧪 Try Example Job"):
    st.session_state["job_description"] = "Earn money from home! No experience needed. Flexible hours. Apply now to start today."
    st.session_state["use_example"] = True
    st.rerun()
job_description = st.text_area(
    "Paste a job description below and find out if it's real or fraudulent.",
    value=st.session_state.get("job_description", ""),
    height=190,
    key="job_desc_input",
    placeholder="Paste the job description here..."
)
model = load_model(model_choice)
explainer = get_explainer(model)

if st.button("🔍 Predict"):
    if job_description.strip() == "":
        st.warning("Please enter a job description to analyze.")
    else:
        result, vectorized = predict_fraudulent(job_description, model)

        # Prediction confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized)[0][1]
            confidence = round(proba * 100, 2)
        else:
            confidence = None

        # Show prediction result
        if result == 1:
            st.error(f"🚨 This job posting is predicted to be **FRAUDULENT**{f' with {confidence}% confidence' if confidence is not None else ''}.")
        else:
            st.success(f"✅ This job posting is predicted to be **REAL**{f' with {100-confidence}% confidence' if confidence is not None else ''}.")

        # Show model used
        st.markdown(f"<div style='margin-top: 1.5em;'><b>Model Used:</b> <span style='color:#F6564D'>{model_choice}</span></div>", unsafe_allow_html=True)

        # Job description statistics
        words = job_description.split()
        word_count = len(words)
        char_count = len(job_description)
        avg_word_len = round(np.mean([len(w) for w in words]), 2) if words else 0
        st.markdown(f"""
        <div style='margin-top: 1.5em;'>
        <b>Job Description Stats:</b><br>
        <ul style='font-size:15px;'>
            <li><b>Word count:</b> {word_count}</li>
            <li><b>Character count:</b> {char_count}</li>
            <li><b>Average word length:</b> {avg_word_len}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Red flag phrase detection
        red_flags = [
            "work from home", "no experience needed", "quick money", "earn money", "flexible hours", "start today", "urgent requirement", "immediate start", "click here", "send your bank details", "investment required", "easy money", "limited positions", "apply now"
        ]
        found_flags = [flag for flag in red_flags if flag in job_description.lower()]
        if found_flags:
            st.markdown(f"<div style='margin-top: 1.5em; color:#F6564D;'><b>Red Flag Phrases Detected:</b> {', '.join(found_flags)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='margin-top: 1.5em; color:#22c55e;'><b>No common red flag phrases detected.</b></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# About Section
# -------------------------------
st.markdown("""
---
### ℹ️ About This App
This app uses a machine learning model (XGBoost or Random Forest) to detect fake job descriptions trained on historical data.
It applies Natural Language Processing (TF-IDF) and model interpretability tools like SHAP for transparency.
""")
