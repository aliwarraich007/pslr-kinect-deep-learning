import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the data
train_file = "../../landmarks_augmented/train.json"
train_labels = "../../labels_augmented/trainLabels.json"
test_file = "../../landmarks_augmented/test.json"
test_labels = "../../labels_augmented/testLabels.json"

with open(train_file, 'r') as json_file:
    training_data = json.load(json_file)
with open(train_labels, 'r') as json_file:
    training_classes = json.load(json_file)
with open(test_file, 'r') as json_file:
    validation_data = json.load(json_file)
with open(test_labels, 'r') as json_file:
    validation_classes = json.load(json_file)

# Convert to numpy arrays and reshape for LSTM input
training_data = np.array(training_data)
training_classes = np.array(training_classes)
training_data = training_data.reshape(training_data.shape[0], 21, 2)

validation_data = np.array(validation_data)
validation_classes = np.array(validation_classes)
validation_data = validation_data.reshape(validation_data.shape[0], 21, 2)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(training_classes)
validation_labels = label_encoder.transform(validation_classes)

# Define the improved LSTM model
rnn_model = Sequential([
    LSTM(256, input_shape=(21, 2), return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(32),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(training_classes)), activation='softmax')
])

rnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
rnn_model.summary()

# Train the model using only training data
history = rnn_model.fit(training_data, encoded_labels, epochs=50, verbose=1)

# Evaluate the model using validation data
loss, accuracy = rnn_model.evaluate(validation_data, validation_labels, verbose=1)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Plot accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], color='blue', linestyle='-', linewidth=2, marker='o', markerfacecolor='red', markersize=5)
plt.title('Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['Train'], loc='upper left')
plt.grid(True)
plt.show()

# Plot loss graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], color='blue', linestyle='-', linewidth=2, marker='o', markerfacecolor='red', markersize=5)
plt.title('Model Loss', fontsize=16)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['Train'], loc='upper left')
plt.grid(True)
plt.show()

# Save the model
rnn_model.save('psl_lstm_aug.keras')