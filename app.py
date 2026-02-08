import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import librosa
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import plotly.express as px
import pandas as pd
import tempfile

# -----------------------------
# 1. Page Config & Theme
# -----------------------------
st.set_page_config(
    page_title="💛 Suicide & Stress Detection Dashboard",
    layout="centered",
    page_icon="🧠"
)

# -----------------------------
# 2. Custom Modern Beige UI
# -----------------------------
st.markdown("""
    <style>
        /* Background */
        body, .stApp {
            background-color: #f9f4e7;
        }

        /* Container styling */
        .main {
            background-color: #f9f4e7;
            padding: 2rem;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #3e2723;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #6d4c41, #8d6e63);
            color: white;
            border-radius: 12px;
            border: none;
            font-size: 16px;
            padding: 10px 25px;
            transition: 0.3s ease;
            box-shadow: 0px 4px 10px rgba(100, 70, 50, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #8d6e63, #a1887f);
            color: #fff9c4;
            transform: scale(1.02);
        }

        /* Text Areas and Inputs */
        .stTextArea textarea, .stTextInput input {
            background-color: #fff8e1 !important;
            border: 1px solid #c7a97f !important;
            border-radius: 10px !important;
            color: #3e2723;
        }

        /* Radio buttons */
        .stRadio label {
            color: #3e2723 !important;
            font-weight: 600;
        }

        /* Metric Cards */
        .stMetric {
            background-color: #fff8e1;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(150, 100, 60, 0.15);
            padding: 10px;
        }

        /* Empathy Box */
        .empathy-box {
            background-color: #fbe9e7;
            padding: 15px;
            border-radius: 12px;
            border-left: 6px solid #d84315;
            margin-top: 20px;
        }

        /* Footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# 3. Model Definition
# -----------------------------
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 10 * 43 + 768, 2)

    def forward(self, mfcc, input_ids, attention_mask):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output
        audio_feat = self.audio_cnn(mfcc.unsqueeze(1))
        audio_feat = audio_feat.view(audio_feat.size(0), -1)
        combined = torch.cat((text_feat, audio_feat), dim=1)
        return self.fc(combined)


@st.cache_resource
def load_model_and_tokenizer():
    model = MultiModalModel()
    model.load_state_dict(torch.load("multimodal_suicide_detection.pth", map_location=torch.device('cpu')))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()


# -----------------------------
# 4. Helper Functions
# -----------------------------
def extract_mfcc(file_path, max_pad_len=174, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

def predict(model, tokenizer, text, mfcc):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=64)
    with torch.no_grad():
        outputs = model(mfcc, enc['input_ids'], enc['attention_mask'])
        probs = torch.softmax(outputs, dim=1).numpy()[0]
    label = np.argmax(probs)
    return label, probs

def interpret_emotion(prob):
    if prob < 0.4:
        return "🟢 Low Risk (Normal)"
    elif 0.4 <= prob < 0.7:
        return "🟠 Moderate Stress"
    else:
        return "🔴 High Suicide Risk"

def empathy_message(risk_label):
    if "High" in risk_label:
        st.markdown("""
        <div class='empathy-box'>
        <strong>💬 You are not alone.</strong><br>
        If you or someone you know is struggling, please reach out for help:<br>
        • 🇮🇳 AASRA Helpline: 91-9820466726<br>
        • 📞 Snehi: 91-9582208181<br>
        • 🌍 International: <a href='https://findahelpline.com', target='_blank'>findahelpline.com</a>
        </div>
        """, unsafe_allow_html=True)

def voice_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""


# -----------------------------
# 5. App Layout
# -----------------------------
st.title("💛 Suicide & Stress Detection Dashboard")
st.markdown("#### Empowering early mental health support through multimodal AI")

option = st.radio("🎯 Choose Input Type:", ["📝 Text", "🎤 Voice"])

# --- TEXT INPUT ---
if option == "📝 Text":
    st.subheader("🗒️ Text-based Detection")
    text_input = st.text_area("Enter your message or journal entry:")
    if st.button("🔍 Analyze Text"):
        if text_input.strip():
            dummy_mfcc = torch.zeros((1, 40, 174))
            label, probs = predict(model, tokenizer, text_input, dummy_mfcc)
            suicide_prob = float(probs[1])
            emotion = interpret_emotion(suicide_prob)
            st.markdown(f"### **Result:** {emotion}")
            empathy_message(emotion)
            df = pd.DataFrame({
                "Category": ["Non-Suicidal", "Potential Suicide/Stress"],
                "Probability": probs * 100
            })
            fig = px.bar(df, x="Category", y="Probability", color="Category",
                         text="Probability", color_discrete_sequence=["#9c6b30", "#a93c3b"],
                         title="Prediction Confidence (Text Input)")
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please type a message to analyze.")

# --- VOICE INPUT ---
if option == "🎤 Voice":
    st.subheader("🎧 Voice-based Detection")
    uploaded_file = st.file_uploader("Upload a voice clip (.wav or .mp3):", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("🪄 Analyze Voice"):
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            AudioSegment.from_file(uploaded_file).export(temp_wav.name, format="wav")
            transcribed_text = voice_to_text(temp_wav.name)
            if transcribed_text:
                st.success(f"**Transcribed Text:** {transcribed_text}")
                mfcc = extract_mfcc(temp_wav.name)
                label, probs = predict(model, tokenizer, transcribed_text, mfcc)
                suicide_prob = float(probs[1])
                emotion = interpret_emotion(suicide_prob)
                st.markdown(f"### **Result:** {emotion}")
                empathy_message(emotion)
                df = pd.DataFrame({
                    "Category": ["Non-Suicidal", "Potential Suicide/Stress"],
                    "Probability": probs * 100
                })
                fig = px.bar(df, x="Category", y="Probability", color="Category",
                             text="Probability", color_discrete_sequence=["#9c6b30", "#a93c3b"],
                             title="Prediction Confidence (Voice Input)")
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not transcribe audio properly.")





st.markdown("<br><center><b>Designed with empathy 💛 by Vinita Barod</b></center>", unsafe_allow_html=True)
