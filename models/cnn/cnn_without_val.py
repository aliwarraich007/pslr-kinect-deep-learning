import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_file = "../../landmarks_raw/train.json"
train_labels = "../../labels/trainLabels.json"
test_file = "../../landmarks_raw/test.json"
test_labels = "../../labels/testLabels.json"

with open(train_file, 'r') as json_file:
    training_data = json.load(json_file)
with open(train_labels, 'r') as json_file:
    training_classes = json.load(json_file)
with open(test_file, 'r') as json_file:
    test_data = json.load(json_file)
with open(test_labels, 'r') as json_file:
    test_classes = json.load(json_file)

training_data = np.array(training_data).reshape(-1, 21, 2, 1)
test_data = np.array(test_data).reshape(-1, 21, 2, 1)
training_classes = np.array(training_classes)
test_classes = np.array(test_classes)

label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(training_classes)
encoded_test_labels = label_encoder.transform(test_classes)

# Reduce the number of classes for better visualization in the confusion matrix
classes_to_show = 5  # Change this number to the desired number of classes to show
unique_labels, counts = np.unique(test_classes, return_counts=True)
most_common_classes = unique_labels[np.argsort(counts)][::-1][:classes_to_show]
mask = np.isin(test_classes, most_common_classes)
reduced_test_data = test_data[mask]
reduced_encoded_test_labels = encoded_test_labels[mask]
reduced_test_classes = test_classes[mask]

cnn_model = Sequential([
    Conv2D(filters=16, kernel_size=(1,1), activation='relu', input_shape=(21,2,1)),
    MaxPooling2D((1,1)),
    Conv2D(32, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Conv2D(64, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Conv2D(128, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(np.unique(training_classes)), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(training_data, encoded_train_labels, epochs=50, verbose=1)
test_loss, test_accuracy = cnn_model.evaluate(reduced_test_data, reduced_encoded_test_labels, verbose=1)
print(f'Test accuracy: {test_accuracy:.2f}')

predictions = cnn_model.predict(reduced_test_data)
predicted_classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(reduced_encoded_test_labels, predicted_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

cnn_model.save('psl_cnn_raw.keras')
