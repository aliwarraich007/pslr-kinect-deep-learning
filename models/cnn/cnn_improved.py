import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_file = "../../landmarks/landmarks_augmented/train.json"
train_labels = "../../landmarks/labels_augmented/trainLabels.json"
test_file = "../../landmarks/landmarks_augmented/test.json"
test_labels = "../../landmarks/labels_augmented/testLabels.json"

# Load the datasets
with open(train_file, 'r') as json_file:
    training_data = json.load(json_file)
with open(train_labels, 'r') as json_file:
    training_classes = json.load(json_file)
with open(test_file, 'r') as json_file:
    test_data = json.load(json_file)
with open(test_labels, 'r') as json_file:
    test_classes = json.load(json_file)

# Reshape and convert data to numpy arrays
training_data = np.array(training_data).reshape(-1, 21, 2, 1)
test_data = np.array(test_data).reshape(-1, 21, 2, 1)
training_classes = np.array(training_classes)
test_classes = np.array(test_classes)

# Encode labels
label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(training_classes)
encoded_test_labels = label_encoder.transform(test_classes)

# Adjusted CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 1), activation='relu', padding='same', input_shape=(21, 2, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 1), padding='same'),

    Conv2D(64, (3, 1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 1), padding='same'),

    Conv2D(128, (3, 1), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 1), padding='same'),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(np.unique(training_classes)), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

history = cnn_model.fit(training_data, encoded_train_labels,
                        validation_data=(test_data, encoded_test_labels),
                        epochs=50, verbose=1,
                        callbacks=[reduce_lr])

test_loss, test_accuracy = cnn_model.evaluate(test_data, encoded_test_labels, verbose=1)
print(f'Test accuracy: {test_accuracy:.2f}')

cnn_model.save('psl_cnn_aug_improved.keras')
with open('training_history_improved.json', 'w') as f:
    json.dump(history.history, f)

predictions = cnn_model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)


cm = confusion_matrix(encoded_test_labels, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
