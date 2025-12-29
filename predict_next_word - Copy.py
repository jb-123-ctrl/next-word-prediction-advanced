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
# LOAD MODEL (choose one)
# -----------------------------
# Uncomment ONE model at a time

MODEL_PATH = "models/lstm_attention_model.h5"
# MODEL_PATH = "models/gru_attention_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# SETTINGS
# -----------------------------
MAX_SEQ_LEN = model.input_shape[1]

# -----------------------------
# FUNCTION: NEXT WORD PREDICTION
# -----------------------------
def predict_next_word(text):
    text = text.lower()
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_SEQ_LEN,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return ""

# -----------------------------
# USER INPUT LOOP
# -----------------------------
while True:
    user_input = input("\nEnter text (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    next_word = predict_next_word(user_input)
    print("👉 Predicted next word:", next_word)
