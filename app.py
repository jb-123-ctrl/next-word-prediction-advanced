import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Next Word Prediction")
st.write("LSTM • GRU • Hybrid LSTM–GRU with Attention")

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_choice = st.selectbox(
    "Choose Model",
    (
        "LSTM + Attention",
        "GRU + Attention",
        "Hybrid LSTM + GRU + Attention"
    )
)

if model_choice == "LSTM + Attention":
    model = load_model("models/lstm_attention_model.h5")
elif model_choice == "GRU + Attention":
    model = load_model("models/gru_attention_model.h5")
else:
    model = load_model("models/hybrid_lstm_gru_attention_model.h5")

MAX_SEQ_LEN = model.input_shape[1]
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

# -----------------------------
# TEMPERATURE SAMPLING
# -----------------------------
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_next_word(text, temperature):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding="pre")

    preds = model.predict(seq, verbose=0)[0]
    idx = sample_with_temperature(preds, temperature)
    return index_to_word.get(idx, "")

# -----------------------------
# UI INPUTS
# -----------------------------
user_input = st.text_input(
    "Enter text",
    placeholder="e.g. to be or not"
)

temperature = st.slider(
    "Temperature (Creativity)",
    min_value=0.2,
    max_value=1.5,
    value=0.7,
    step=0.1
)

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input, temperature)
        st.success(f"👉 Predicted next word: **{next_word}**")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built by Jayabharathi • Advanced NLP Project")
