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
import random
import docx
import PyPDF2

# -------------------------------
# Custom CSS Styling
# -------------------------------
def local_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
            html, body, [class*="css"], .container, .navbar, .navbar *, .hero, .hero *, .about-section, .about-section *, .choose-model-label, .stTextArea textarea, textarea, .stButton > button, .stButton button, button, .stSelectbox, .stSelectbox *, .stMarkdown, .stAlert, .navbar-title, .navbar-links, .navbar-links *, .navbar-right, .navbar-left, .signup-btn, .hero-title, .hero-subtitle, .result-block, .result-block *, .main-heading {
                font-family: 'Source Sans 3', 'Source Sans Pro', 'Source Sans', Arial, sans-serif !important;
            }
            html, body, [class*="css"]  {
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
                padding-left: 20px;
                padding-right: 20px;
            }
            .navbar {
                width: 100%;
                max-width: auto;
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
                width: 100%;
            }
            .hero-content {
                max-width: auto;
                text-align: left;
            }
            .hero-title {
                font-size: 3.2rem;
                font-weight: 700;
                color: #062659;
                margin-bottom: 0.5rem;
                line-height: 1.1;
            }
            .hero-highlight {
                color: #F6564D;
                font-weight: 700;
                display: inline-block;
                border-bottom: 5px solid #F6564D;
                margin-bottom: 0.5rem;
            }
            .hero-subtitle {
                font-size: 1.25rem;
                color: #062659;
                margin-bottom: 2rem;
            }
            .stSelectbox > div, .stTextArea textarea, .stButton > button {
                font-family: sans-serif !important;
            }
            .stButton > button {
                max-width: 180px;
                min-width: 120px;
                width: 100%;
                margin-left: 0;
                margin-right: 0;
                display: block;
                background: #F6564D;
                color: #fff;
                border: none;
                font-weight: 700;
            }
            .stButton > button:hover {
                background: #d13a2f;
                color: #fff;
                box-shadow: 0 4px 16px rgba(246,86,77,0.12);
            }
            .about-section {
                font-family: sans-serif !important;
            }
            .result-block, .result-block * {
                font-family: sans-serif !important;
            }
            .poppins-heading {
                font-weight: 800 !important;
                color: #062659;
            }
            .main-heading {
                font-size: 28px !important;
                font-weight: 700;
                color: #062659;
                margin-bottom: 0.5rem;
                margin-top: 2.0rem;
            }
            @media (max-width: 900px) {
                .container, .navbar, .hero {
                    max-width: 100% !important;
                    padding-left: 8px !important;
                    padding-right: 8px !important;
                }
                .navbar {
                    flex-direction: column;
                    align-items: flex-start;
                    padding: 12px 0 8px 0;
                }
                .navbar-left {
                    width: 100%;
                    flex-direction: row;
                    align-items: center !important;
                    gap: 8px;
                }
                .navbar-title {
                    width: 100%;
                    text-align: left !important;
                    margin-bottom: 8px;
                    font-size: 1.1rem !important;
                }
                .navbar-right {
                    margin-top: 0;
                    width: 100%;
                    justify-content: flex-start;
                    flex-direction: column;
                    align-items: flex-start !important;
                }
                .navbar-links {
                    width: 100%;
                    justify-content: flex-start;
                    text-align: left;
                    margin-bottom: 8px;
                }
                .signup-btn {
                    margin-left: 0;
                }
                .hero {
                    flex-direction: column;
                    align-items: flex-start;
                }
                .hero img {
                    max-width: 320px !important;
                    width: 100%;
                    margin-top: 12px;
                }
                .choose-model-label {
                    font-size: 1.1rem !important;
                }
            }
            .stats-text {
                font-size: 20px !important;
                font-family: 'Source Sans 3', 'Source Sans Pro', 'Source Sans', Arial, sans-serif !important;
                font-weight: 600;
            }
            .stats-text, .stats-text li {
                font-size: 20px !important;
                font-family: 'Source Sans 3', 'Source Sans Pro', 'Source Sans', Arial, sans-serif !important;
                font-weight: 600 !important;
            }
            /* Set font size for Streamlit expander headers in prediction history */
            .st-expander > label, .st-expanderHeader {
                font-size: 20px !important;
            }
            /* Set font size for <p>Prediction #N (FRAUDULENT)</p> if used */
            p.prediction-label {
                font-size: 20px !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Set page layout and apply custom styles
st.set_page_config(page_title="Fake Job Detector", layout="wide")
local_css()

# Load vectorizer
try:
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error(f"❌ Failed to load vectorizer: {e}")
    st.stop()

# Load model
def load_model(model_name):
    try:
        if model_name == "XGBoost":
            return joblib.load("xgb_fraud_detector.pkl")
        elif model_name == "Random Forest":
            return joblib.load("job_post_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\\S+|www\\.\\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\w*\\d\\w*', '', text)
    return text

def predict_fraudulent(job_post_text, model):
    cleaned = clean_text(job_post_text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)
    return prediction[0], vect

def get_explainer(model):
    return shap.TreeExplainer(model)

def explain_prediction(text, model, explainer):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    shap_values = explainer.shap_values(vect.toarray(), check_additivity=False)
    features = vectorizer.get_feature_names_out()
    if isinstance(shap_values, list):
        scores = shap_values[1][0]
    else:
        scores = shap_values[0]
    top_indices = np.argsort(np.abs(scores))[::-1][:10]
    return [(features[i], float(scores[i])) for i in top_indices]

# Start the container and place all content inside
st.markdown("""
<div class='container'>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
  <div class="navbar">
      <div class="navbar-left">
          <span class="navbar-logo"><img src="https://img.icons8.com/ios-filled/50/000000/hacker.png" alt="Logo"></span>
          <span class="navbar-title">
              Fake Job Detector
          </span>
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
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
  <div class="hero">
      <div class="hero-content">
          <div class="hero-title">Fake Job Detector<br><span class="hero-highlight">stop wasting your time!</span></div>
          <div class="hero-subtitle">Spot fake job offers instantly with AI-powered accuracy.</div>
      </div>
      <div>
          <img src="https://fastcompanyme.com/wp-content/uploads/2023/03/Why-are-employment-scams-on-the-rise-in-MEA.jpeg" 
               alt="Fake Job Illustration" 
               style="max-width: 480px; margin-top: 32px; border-radius: 18px; box-shadow: 0 4px 24px rgba(6,38,89,0.10);">
      </div>
  </div>
""", unsafe_allow_html=True)

# Form fields
st.markdown('<div class="main-heading">Choose Model</div>', unsafe_allow_html=True)
model_choice = st.selectbox("", ["XGBoost", "Random Forest"], key="choose-model")
fake_job_examples = [
    "Earn money from home! No experience needed. Flexible hours. Apply now to start today.",
    "Congratulations! You have been selected for a high-paying job. Send your bank details to claim your offer.",
    "Work just 2 hours a day and make $5000 a week! Limited positions available, sign up now!",
    "Immediate start! No interview required. Click here to begin earning instantly.",
    "Investment required to secure your job. Double your money in a month! Contact us now."
]
if st.button("Try Example Job"):
    st.session_state["job_description"] = random.choice(fake_job_examples)
    st.session_state["use_example"] = True
    st.rerun()

# File uploader for job descriptions
st.markdown('<div class="main-heading">Upload a job description (.txt, .docx, .pdf)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    " ",
    type=["txt", "docx", "pdf"],
    label_visibility="collapsed"
)
file_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_text = ""
        for page in pdf_reader.pages:
            file_text += page.extract_text() or ""
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        doc = docx.Document(uploaded_file)
        file_text = "\n".join([para.text for para in doc.paragraphs])
    if file_text:
        st.session_state["job_description"] = file_text

# Add a large, bold label above the text area
st.markdown('<div class="main-heading">Paste a job description below and find out if it\'s real or fraudulent.</div>', unsafe_allow_html=True)
job_description = st.text_area(
    "",
    value=st.session_state.get("job_description", ""),
    height=300,
    key="job_desc_input",
    placeholder="Paste the job description here...",
    help=" ",  # hack to allow for custom spacing below
)
st.markdown('<div style="margin-bottom:2.2rem;"></div>', unsafe_allow_html=True)

# After job_description is set, show summary only

if job_description.strip():
    # Simple extractive summary: first 2-3 sentences
    import re
    sentences = re.split(r'(?<=[.!?]) +', job_description.strip())
    summary = ' '.join(sentences[:3]) if len(sentences) > 1 else sentences[0]
    st.markdown('<div class="main-heading">Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.05rem; background:#f3f4f6; border-radius:8px; padding:0.5rem; margin-top:1.0rem;">{summary}</div>', unsafe_allow_html=True)

model = load_model(model_choice)
explainer = get_explainer(model)

# Initialize prediction history in session state
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if st.button("Predict"):
    if job_description.strip() == "":
        st.warning("Please enter a job description to analyze.")
    else:
        st.markdown("<div class='result-block'>", unsafe_allow_html=True)
        result, vectorized = predict_fraudulent(job_description, model)
        proba = model.predict_proba(vectorized)[0][1] if hasattr(model, "predict_proba") else None
        confidence = round(proba * 100, 2) if proba else None
        # Store prediction in history
        st.session_state["prediction_history"].append({
            "job_description": job_description,
            "model": model_choice,
            "result": "FRAUDULENT" if result == 1 else "REAL",
            "confidence": confidence
        })
        if result == 1:
            st.markdown(f"""
            <div style='background:#FEE2E2;padding:1rem 1.5rem;border-radius:10px;color:#B91C1C;font-size:1.2rem;font-weight:700;margin-bottom:1.5rem;'>
                🚨 <b>This job posting is predicted to be <span style='color:#B91C1C;'>FRAUDULENT</span></b> with {confidence}% confidence.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#DCFCE7;padding:1rem 1.5rem;border-radius:10px;color:#15803D;font-size:1.2rem;font-weight:700;margin-bottom:1.5rem;'>
                ✅ <b>This job posting is predicted to be <span style='color:#15803D;'>REAL</span></b> with {100-confidence}% confidence.
            </div>
            """, unsafe_allow_html=True)

        # Confidence Progress Bar
        conf_val = confidence if result == 1 else 100-confidence
        if conf_val >= 80:
            bar_color = '#22c55e'  # green
        elif conf_val >= 50:
            bar_color = '#f59e42'  # orange
        else:
            bar_color = '#ef4444'  # red
        st.markdown(f"""
        <div style='margin-bottom:2.2rem;'>
            <div style='font-size:24px; font-weight:800; margin-bottom:0.4rem;'>Confidence : </div>
            <div style='background:#f3f4f6; border-radius:12px; height:32px; width:100%; max-width:500px; position:relative;'>
                <div style='background:{bar_color}; width:{conf_val}%; height:100%; border-radius:12px; transition:width 0.5s;'></div>
                <div style='position:absolute; left:50%; top:0; height:100%; display:flex; align-items:center; justify-content:center; font-size:1.3rem; font-weight:800; color:#222; transform:translateX(-50%);'>{conf_val}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='main-heading' style='font-size:24px !important;'>Model :  <span style='color:#F6564D; font-weight:700; font-size:24px !important;'> {model_choice}</span></div>", unsafe_allow_html=True)

        words = job_description.split()
        st.markdown(f"""
        <div style='margin-top:1.5rem; margin-bottom:1.5rem;'>
        <div style='font-size:1.5rem; font-weight:800; color:#062659; margin-bottom:0.5rem;'>Job Description Stats:</div>
        """ + f"""
        <ul class='stats-text'>
            <li class='stats-text'><b>Word count:</b> {len(words)}</li>
            <li class='stats-text'><b>Character count:</b> {len(job_description)}</li>
            <li class='stats-text'><b>Average word length:</b> {round(sum(len(w) for w in words)/len(words), 1) if words else 0}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        # Improved red flag detection (case-insensitive, punctuation/whitespace-insensitive, robust to newlines)
        import re
        red_flags = [
    "work from home", "no experience needed", "quick money", "earn money", "flexible hours", 
    "start today", "urgent requirement", "immediate start", "click here", "send your bank details", 
    "investment required", "easy money", "limited positions", "apply now", "cash daily", 
    "make $$$ fast", "100% legit", "training provided", "guaranteed income", "no interview", 
    "get rich", "reshipping", "wire transfer", "bitcoin", "crypto job", "payment upfront", 
    "short hours, high pay", "be your own boss", "hiring immediately", "quick approval", 
    "minimal work", "paid to sign up", "signing bonus", "exclusive opportunity", 
    "just your phone required", "data entry from home", "envelope stuffing", 
    "free training", "limited time only", "send CV to WhatsApp", "Skype interview only", 
    "no resume needed", "commission only", "multi-level marketing", "MLM", 
    "pay to apply", "processing fees", "job guarantee", "lucrative opportunity", 
    "no background check", "freelance scam", "we pay you to train", "secret shopper", 
    "mystery shopper", "foreign representative", "remote assistant", "join our team today"
    ]

        cleaned_desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', job_description.lower())
        cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc)  # collapse all whitespace
        found_flags = []
        for flag in red_flags:
            # Allow for any whitespace (including newlines) between words in the flag
            pattern = r'\b' + r'\s+'.join(map(re.escape, flag.split())) + r'\b'
            if re.search(pattern, cleaned_desc, re.IGNORECASE):
                found_flags.append(flag)
        if found_flags:
            st.markdown("""
            <div style='margin-top:2rem; font-size:24px; font-weight:800; color:#F6564D;'>Red Flag Phrases Detected:</div>
            <ul class='stats-text' style='color:#F6564D; font-weight:600; margin-top:0.5rem;'>
            """ + "\n".join([f"<li class='stats-text'>{flag}</li>" for flag in found_flags]) + "</ul>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='margin-top:2rem; color:#22c55e; font-size:1.35rem; font-weight:800;'>No common red flag phrases detected.</div>", unsafe_allow_html=True)

# Display prediction history
if st.session_state["prediction_history"]:
    st.markdown('<div class="main-heading">Prediction History</div>', unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state["prediction_history"]), 1):
        with st.expander(f"Prediction #{len(st.session_state['prediction_history']) - i + 1} ({entry['result']})"):
            st.write(f"**Model:** {entry['model']}")
            st.write(f"**Confidence :** {entry['confidence']}%")
            st.write("**Job Description:**")
            st.write(entry["job_description"])

# About section
st.markdown('<div class="main-heading">About This App</div>', unsafe_allow_html=True)
st.markdown("""
<div class='about-section'>

**Fake Job Detector** is an AI-powered tool designed to help job seekers and recruiters quickly identify potentially fraudulent job postings. Simply paste a job description and our app will analyze it using advanced machine learning models to determine if it is likely to be real or fake.

**How it works:**
- Uses Natural Language Processing (NLP) to extract features from job descriptions.
- Employs two powerful models: XGBoost and Random Forest, trained on a large dataset of real and fake job postings.
- Highlights suspicious phrases and provides statistics about your input.
- Offers transparency by showing which model was used and the confidence of the prediction.

**Who is this for?**
- Job seekers who want to avoid scams and protect themselves online.
- Recruiters and HR professionals who want to screen postings for authenticity.
- Anyone interested in AI-powered fraud detection in the job market.

**Responsible Use:**
- This tool is intended as an aid, not a guarantee. Always use your own judgment and verify job offers through official channels.
- The app does not store or share your input data.

**Built with:** Streamlit, Python, scikit-learn, XGBoost, SHAP, and a passion for safer job searching!
</div>
""", unsafe_allow_html=True)
