import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import json

# Loading the data from JSON 
with open('data.json') as json_file:
  data = json.load(json_file)

inputs = [item["input"] for item in data]
labels = [item["output"] for item in data]

# Text Tokenization
tokenizer = Tokenizer(num_words=5000) 
tokenizer.fit_on_texts(inputs)
sequences = tokenizer.texts_to_sequences(inputs)

# Padding the Sequences 
max_len = 100  
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Label encoding 
label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(labels)
one_hot_labels = keras.utils.to_categorical(encoded_labels)
num_categories = len(label_encoder.classes_)  

# Building the Model here
model = keras.Sequential([
  keras.layers.Embedding(5000, 128, input_length=max_len),  # Adjust embedding_dim as needed
  keras.layers.LSTM(64),  # Experiment with different hidden layers (LSTM or Dense)
  keras.layers.Dense(num_categories, activation='softmax')
])

# Here we are compiling the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model with data
model.fit(padded_sequences, one_hot_labels, epochs=15)

# Prediction on data
new_text = "choclate chip cookie" 
new_sequence = tokenizer.texts_to_sequences([new_text])
padded_new_sequence = pad_sequences(new_sequence, maxlen=max_len)

prediction = model.predict(padded_new_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
print(f"Predicted label for '{new_text}': {predicted_label}")