import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download

# ---------------------
# Load Files From HuggingFace
# ---------------------
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

# ---------------------
# UI
# ---------------------
st.title("🧠 Silent Sentinel - Suicide Detection")

user_text = st.text_area("Enter text")

# ---------------------
# Prediction
# ---------------------
def predict(text):

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        return "⚠ High Suicide Risk"
    else:
        return "✅ Low Risk"

if st.button("Predict"):

    if user_text:
        result = predict(user_text)
        st.success(result)
    else:
        st.warning("Please enter text")
