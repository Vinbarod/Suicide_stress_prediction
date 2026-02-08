import streamlit as st
import torch
from huggingface_hub import hf_hub_download

# -------- Load Model --------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/multimodal-suicide-detection",
        filename="multimodal_model.pth"
    )
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

model = load_model()

# -------- UI --------
st.title("🧠 Silent Sentinel - Suicide & Stress Detection")

st.write("Enter text or upload audio for mental health risk prediction")

# -------- Input --------
user_text = st.text_area("Enter Text")

# -------- Prediction Function --------
def predict(text):
    # Replace with your real preprocessing & model logic
    if len(text) > 20:
        return "High Risk"
    else:
        return "Low Risk"

# -------- Button --------
if st.button("Predict"):
    if user_text:
        result = predict(user_text)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter text")
