import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, GRU, Dense,
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------
# CPU SAFETY (Windows stability)
# -----------------------------------
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# -----------------------------------
# LOAD DATA
# -----------------------------------
X = np.load("X.npy", mmap_mode="r")
y = np.load("y.npy", mmap_mode="r")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

total_words = y.shape[1]
max_seq_len = X.shape[1]

# -----------------------------------
# HYBRID MODEL
# -----------------------------------
inputs = Input(shape=(max_seq_len,))

# Embedding
x = Embedding(
    input_dim=total_words,
    output_dim=128,
    input_length=max_seq_len
)(inputs)

# LSTM layer (long-term memory)
x = LSTM(128, return_sequences=True)(x)

# GRU layer (short-term refinement)
x = GRU(128, return_sequences=True)(x)

# Attention layer
attn = Attention()([x, x])

# Reduce sequence to vector
context = GlobalAveragePooling1D()(attn)

# Output
outputs = Dense(total_words, activation="softmax")(context)

# Build model
model = Model(inputs, outputs)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# TRAINING
# -----------------------------------
early_stop = EarlyStopping(
    monitor="loss",
    patience=2,
    restore_best_weights=True
)

model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# Save model
model.save("models/hybrid_lstm_gru_attention_model.h5")
print("✅ HYBRID LSTM + GRU + Attention model trained & saved")
