import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# Loading the data from JSON here
with open('data.json') as json_file:
    data = json.load(json_file)

inputs = [item["input"] for item in data]
labels = [item["output"] for item in data]

# Splitting the data into (training, validation, and test)
X_train, X_temp, y_train, y_temp = train_test_split(inputs, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Text Tokenization here
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_val = tokenizer.texts_to_sequences(X_val)
sequences_test = tokenizer.texts_to_sequences(X_test)

# Padding the Sequences 
max_len = 5
padded_sequences_train = pad_sequences(sequences_train, maxlen=max_len)
padded_sequences_val = pad_sequences(sequences_val, maxlen=max_len)
padded_sequences_test = pad_sequences(sequences_test, maxlen=max_len)

# Label encoding
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_labels_train = label_encoder.transform(y_train)
encoded_labels_val = label_encoder.transform(y_val)
encoded_labels_test = label_encoder.transform(y_test)
num_categories = len(label_encoder.classes_)

# Building the Model (Embedding still too large?)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=1100, output_dim=128),
    keras.layers.LSTM(64),
    keras.layers.Dense(num_categories, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the Model
model.fit(padded_sequences_train, encoded_labels_train, epochs=30, validation_data=(padded_sequences_val, encoded_labels_val))

# Evaluating the Model on Test Set
test_loss, test_accuracy = model.evaluate(padded_sequences_test, encoded_labels_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Prediction on a new text
new_text = "ground beef"
new_sequence = tokenizer.texts_to_sequences([new_text])
padded_new_sequence = pad_sequences(new_sequence, maxlen=max_len)

prediction = model.predict(padded_new_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
print(f"Predicted label for '{new_text}': {predicted_label}")