import nltk
from nltk.corpus import gutenberg
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Download dataset
nltk.download('gutenberg')

# Load Hamlet text
text = gutenberg.raw('shakespeare-hamlet.txt')

# Clean text
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

# Save cleaned text
with open("data/hamlet.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total words:", total_words)

# Create sequences
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

# Padding
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Split X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Save outputs
np.save("X.npy", X)
np.save("y.npy", y)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Preprocessing completed successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
