import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the data
train_file = "../../landmarks/landmarks_augmented/train.json"
train_labels = "../../landmarks/labels_augmented/trainLabels.json"
test_file = "../../landmarks/landmarks_augmented/test.json"
test_labels = "../../landmarks/labels_augmented/testLabels.json"

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
    Bidirectional(LSTM(256, return_sequences=True), input_shape=(21, 2)),
    BatchNormalization(),
    Dropout(0.5),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.5),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(training_classes)), activation='softmax')
])

rnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
rnn_model.summary()

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model using training and validation data
history = rnn_model.fit(training_data, encoded_labels,
                        validation_data=(validation_data, validation_labels),
                        epochs=50, verbose=1,
                        callbacks=[reduce_lr])

# Evaluate the model using validation data
loss, accuracy = rnn_model.evaluate(validation_data, validation_labels, verbose=1)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss', fontsize=16)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

predictions = rnn_model.predict(validation_data)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(validation_labels, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save the model
rnn_model.save('psl_lstm_aug_improved.keras')
