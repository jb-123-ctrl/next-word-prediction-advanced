import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# LOAD MODEL (CHOOSE ONE)
# -----------------------------
MODEL_PATH = "models/hybrid_lstm_gru_attention_model.h5"
# MODEL_PATH = "models/lstm_attention_model.h5"
# MODEL_PATH = "models/gru_attention_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

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
# PREDICT NEXT WORD
# -----------------------------
def predict_next_word(text, temperature=1.0):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding="pre")

    preds = model.predict(seq, verbose=0)[0]
    idx = sample_with_temperature(preds, temperature)
    return index_to_word.get(idx, "")

# -----------------------------
# INTERACTIVE LOOP
# -----------------------------
print("\nTemperature guide: 0.3 (safe) | 0.7 (balanced) | 1.2 (creative)")
while True:
    text = input("\nEnter text (or 'exit'): ")
    if text.lower() == "exit":
        break

    temp = float(input("Temperature: "))
    word = predict_next_word(text, temp)
    print("👉 Next word:", word)
