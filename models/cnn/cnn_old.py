import json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import numpy as np
train_file = "../../landmarks_raw/train.json"
train_labels = "../../labels_raw/trainLabels.json"
test_file = "../../landmarks_raw/test.json"
test_labels = "../../labels_raw/testLabels.json"
with open(train_file, 'r') as json_file:
    training_data = json.load(json_file)
with open(train_labels, 'r') as json_file:
    training_classes = json.load(json_file)
with open(test_file, 'r') as json_file:
    validation_data = json.load(json_file)
with open(test_labels, 'r') as json_file:
    validation_classes = json.load(json_file)
training_data=np.array(training_data)
training_classes=np.array(training_classes)
training_data = training_data.reshape(training_data.shape[0], 21, 2, 1)
validation_data=np.array(validation_data)
validation_classes=np.array(validation_classes)
validation_data = validation_data.reshape(validation_data.shape[0], 21, 2, 1)
num_classes = len(np.unique(training_classes))  
# cnn_model = Sequential([
#     Conv2D(32, (1, 1), activation='relu', input_shape=(21, 2, 1)),
#     MaxPooling2D((1, 1)),
#     Conv2D(64, (1, 1), activation='relu'),
#     MaxPooling2D((1, 1)),
#     Conv2D(128, (1, 1), activation='relu'),
#     MaxPooling2D((1, 1)),
#     Flatten(),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dense(num_classes, activation='softmax')
# ])
cnn_model = Sequential([
    Conv2D(64, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Conv2D(128, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Conv2D(256, (1, 1), activation='relu'),
    MaxPooling2D((1, 1)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
cnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(training_classes)
history=cnn_model.fit(training_data, encoded_labels, epochs=50,validation_data=(validation_data, label_encoder.fit_transform(validation_classes)), verbose=1)
loss, accuracy = cnn_model.evaluate(validation_data, label_encoder.fit_transform(validation_classes), verbose=1)

print(f'Test accuracy: {accuracy:.2f}')
# Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

cnn_model.save('psl_cnn_aug_improved.keras')
