import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Silent Sentinel Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():

    model_path = hf_hub_download(
        repo_id="Veenita/silent-sentinel-model",
        filename="model.h5"
    )

    tokenizer_path = hf_hub_download(
        repo_id="Veenita/silent-sentinel-model",
        filename="tokenizer.pkl"
    )

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_artifacts()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Silent Sentinel")
st.sidebar.info("AI Mental Health Risk Detection")
st.sidebar.write("Developed by Vinita Barod")

st.sidebar.divider()

st.sidebar.warning(
    "⚠ This tool provides awareness support and is not a medical diagnosis."
)

# ---------------- HEADER ----------------
st.title("🧠 Silent Sentinel Dashboard")
st.caption("AI Powered Suicide & Stress Risk Monitoring")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area("💬 Share your thoughts or feelings")

with col2:
    st.info("Write freely. Your data is not stored.")

# ---------------- PREDICTION FUNCTION ----------------
def predict(text):

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded)[0][0]
    return prediction

# ---------------- SUGGESTION ENGINE ----------------
def get_suggestions(score):

    if score > 0.7:
        return "High Risk", [
            "Talk to someone you trust",
            "Seek professional counseling",
            "Avoid isolation",
            "Call mental health helpline"
        ]

    elif score > 0.4:
        return "Moderate Stress", [
            "Practice meditation",
            "Maintain sleep routine",
            "Engage in hobbies",
            "Stay socially connected"
        ]

    else:
        return "Low Risk", [
            "Maintain healthy lifestyle",
            "Continue social engagement",
            "Practice gratitude journaling"
        ]

# ---------------- ANALYZE BUTTON ----------------
if st.button("🔍 Analyze Mental Health"):

    if user_text:

        score = predict(user_text)
        level, tips = get_suggestions(score)

        st.divider()

        # -------- METRICS --------
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Risk Score", f"{score*100:.2f}%")

        with m2:
            st.metric("Risk Level", level)

        with m3:
            st.metric("Analysis Time", datetime.now().strftime("%H:%M:%S"))

        # -------- PROGRESS BAR --------
        st.subheader("📊 Risk Probability")
        st.progress(float(score))

        # -------- EMOTION CHART --------
        st.subheader("📈 Emotional Pattern Analysis")

        emotions = ["Stress", "Depression", "Anxiety", "Neutral"]
        values = [
            score * 0.9,
            score * 0.8,
            score * 0.7,
            1 - score
        ]

        fig = plt.figure()
        plt.bar(emotions, values)
        plt.title("Emotional Indicators")
        st.pyplot(fig)

        # -------- SUGGESTIONS --------
        st.subheader("💡 Personalized Suggestions")

        for tip in tips:
            st.write(f"✔ {tip}")

        # -------- HELPLINE --------
        if score > 0.7:
            st.error("☎ India Mental Health Helpline: 1800-599-0019")

        # -------- REPORT DOWNLOAD --------
        report = pd.DataFrame({
            "Input Text": [user_text],
            "Risk Score": [score],
            "Risk Level": [level]
        })

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📄 Download Report",
            csv,
            "silent_sentinel_report.csv",
            "text/csv"
        )

    else:
        st.warning("Please enter text for analysis")
