# 🛡️ Fake Job Detector using Machine Learning

A real-time web application that detects fraudulent job postings using machine learning and NLP techniques. Designed especially for students and job seekers to stay protected from employment scams.

## 🔍 Project Overview

Scams through fake job offers are increasingly common online. This project uses supervised machine learning algorithms (XGBoost, Random Forest) to classify job descriptions as **real** or **fraudulent**.

The model is deployed via a modern **Streamlit** web interface, allowing users to:
- Paste or type job descriptions
- Select the model for prediction
- Instantly detect potential frauds
- Understand why a job was flagged using SHAP explainability
- See red-flag keywords and job posting stats

---

## 💡 Key Features

- ✍️ Clean and responsive UI using **custom Streamlit CSS**
- 🧠 **XGBoost & Random Forest** for robust predictions
- 📊 **TF-IDF** vectorization for feature extraction
- 🔎 SHAP explanations for model transparency
- 🚩 Red flag keyword detection
- 🔁 Try example job descriptions instantly
- 📜 Prediction insights: word count, character count, etc.
- 💾 Upcoming: Prediction history

---

## 🛠️ Tech Stack

| Category        | Tech Used                        |
|----------------|----------------------------------|
| Language        | Python 3.11                      |
| Libraries       | scikit-learn, XGBoost, SHAP, Streamlit |
| Frontend UI     | Streamlit + Custom CSS           |
| NLP             | TF-IDF (via scikit-learn)        |
| Deployment      | Streamlit Cloud / Ngrok (for testing) |
| Model Format    | joblib `.pkl`                    |

---

## 🚀 Getting Started

### 🔧 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/Abhi-raj-R/fake-job-detector.git
cd fake-job-detector
