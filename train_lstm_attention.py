import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense,
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

total_words = y.shape[1]
max_seq_len = X.shape[1]

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
inputs = Input(shape=(max_seq_len,))

# Embedding layer
x = Embedding(
    input_dim=total_words,
    output_dim=128,
    input_length=max_seq_len
)(inputs)

# LSTM layer (return sequences for attention)
lstm_out = LSTM(128, return_sequences=True)(x)

# Attention layer
attention_out = Attention()([lstm_out, lstm_out])

# Reduce sequence to vector
context_vector = GlobalAveragePooling1D()(attention_out)

# Output layer
outputs = Dense(total_words, activation="softmax")(context_vector)

# Build model
model = Model(inputs, outputs)

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAINING
# -----------------------------
early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X, y,
    epochs=20,
    batch_size=64,
    callbacks=[early_stop]
)

# Save model
model.save("models/lstm_attention_model.h5")
print("✅ Attention-based LSTM model saved")
