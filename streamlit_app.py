import streamlit as st
import json
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from collections import Counter
import tempfile
from gtts import gTTS
import time
import os
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(layout="wide")
genai.configure(api_key="AIzaSyAFr_rz56WYhxevObkmx1w2FNYFvPmRPPY")

# =====================================================
# GLOWING TITLE
# =====================================================

st.markdown("""
<style>
.glow {
  font-size:40px;
  color:#00f;
  text-align:center;
  animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
  from { text-shadow: 0 0 10px #00f; }
  to { text-shadow: 0 0 20px #0ff; }
}
.chat-user {
  background-color:#DCF8C6;
  padding:10px;
  border-radius:10px;
  margin:5px;
}
.chat-ai {
  background-color:#F1F0F0;
  padding:10px;
  border-radius:10px;
  margin:5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glow">🤖 Jabez Emotional AI</div>', unsafe_allow_html=True)

# =====================================================
# DARK/LIGHT MODE
# =====================================================

theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    body { background-color:#111; color:white; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# LOAD DATASET
# =====================================================

with open("dataset.json", "r", encoding="utf-8") as f:
    memory_data = json.load(f)

if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

if "chat" not in st.session_state:
    st.session_state.chat = []

# =====================================================
# EMOTION DETECTION (ADVANCED)
# =====================================================

emotion_keywords = {
    "joy": ["happy", "great", "excited"],
    "sadness": ["sad", "cry", "hurt", "miss"],
    "anxiety": ["worried", "anxious", "scared"],
    "frustration": ["angry", "hate", "annoyed"],
    "burnout": ["tired", "exhausted", "drained"],
    "affection": ["love", "care", "miss you"],
    "motivated": ["achieve", "win", "success"],
}

def detect_emotion(text):
    text = text.lower()
    for emotion, words in emotion_keywords.items():
        for w in words:
            if w in text:
                return emotion
    return "neutral"

# =====================================================
# EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

def flatten_memory(data):
    texts = []
    for section in data.values():
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    texts.extend([str(v) for v in item.values()])
                else:
                    texts.append(str(item))
        elif isinstance(section, dict):
            texts.extend([str(v) for v in section.values()])
    return texts

memory_texts = flatten_memory(memory_data)

def retrieve_context(query):
    embeddings = embed_model.encode(memory_texts, convert_to_tensor=True)
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:3]
    return [memory_texts[i] for i in top_idx]

# =====================================================
# RESPONSE LENGTH CONTROL
# =====================================================

def length_instruction(emotion):
    if emotion in ["sadness", "burnout"]:
        return "Respond in 120 words with gentle tone."
    if emotion in ["joy", "motivated"]:
        return "Respond in 70 words energetic."
    if emotion in ["anxiety", "frustration"]:
        return "Respond in 100 words calm and grounding."
    return "Respond in 60 words natural."

# =====================================================
# CHAT UI
# =====================================================

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f'<div class="chat-user">🧍 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-ai">🤖 {msg}</div>', unsafe_allow_html=True)

user_input = st.text_input("Talk to Jabez")

if st.button("Send") and user_input:

    st.session_state.chat.append(("user", user_input))

    emotion = detect_emotion(user_input)
    st.session_state.emotion_log.append({
        "emotion": emotion,
        "time": datetime.now()
    })

    context = "\n".join(retrieve_context(user_input))
    instruction = length_instruction(emotion)

    prompt = f"""
You are Jabez AI.
Maintain ethical boundaries.
Do not create dependency.

Detected emotion: {emotion}

{instruction}

Memory Context:
{context}

User: {user_input}
Jabez:
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    ai_text = response.text

    st.session_state.chat.append(("ai", ai_text))

# =====================================================
# EMOTION ANALYTICS DASHBOARD
# =====================================================

st.sidebar.markdown("## 📊 Emotion Analytics")

if st.sidebar.button("Show Analytics"):

    emotions = [e["emotion"] for e in st.session_state.emotion_log]
    counts = Counter(emotions)

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set_title("Emotion Frequency")
    st.pyplot(fig)

    negative = counts.get("sadness",0) + counts.get("anxiety",0) + counts.get("burnout",0)
    total = sum(counts.values())

    if total > 5 and negative/total > 0.5:
        st.warning("⚠️ You've shown high emotional strain recently.")
        st.info("Suggestion: Take a short walk, drink water, or talk to someone you trust.")

# =====================================================
# SELF REFLECTION MODE
# =====================================================

if st.sidebar.button("🧠 AI Self Reflection"):

    if st.session_state.emotion_log:
        last = st.session_state.emotion_log[-1]["emotion"]
        st.write("Last detected emotion:", last)
        st.write("Response style adapted based on emotion.")
        st.write("Memory context was retrieved using embeddings.")
        st.write("Ethical boundary: Dependency prevention active.")
