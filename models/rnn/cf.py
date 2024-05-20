import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_path = 'psl_lstm_aug.keras'
rnn_model = load_model(model_path)

# Load the validation data
test_file = "../../landmarks_augmented/test.json"
test_labels = "../../labels_augmented/testLabels.json"

with open(test_file, 'r') as json_file:
    validation_data = json.load(json_file)
with open(test_labels, 'r') as json_file:
    validation_classes = json.load(json_file)

# Convert to numpy arrays and reshape for LSTM input
validation_data = np.array(validation_data)
validation_classes = np.array(validation_classes)
validation_data = validation_data.reshape(validation_data.shape[0], 21, 2)

# Encode labels
label_encoder = LabelEncoder()
validation_labels = label_encoder.fit_transform(validation_classes)

# Predict the labels for validation data
validation_predictions = rnn_model.predict(validation_data)
validation_predictions_labels = np.argmax(validation_predictions, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(validation_labels, validation_predictions_labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(20, 20))  # Increase the figure size

# Customize the colormap to highlight the diagonal
cmap = sns.color_palette("light:#5A9_r", as_cmap=True)

# Plot the heatmap
ax = sns.heatmap(conf_matrix, annot=False, fmt='d', cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
plt.xlabel('Predicted Labels', fontsize=16)
plt.ylabel('True Labels', fontsize=16)
plt.title('Confusion Matrix', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=14)  # Rotate x-tick labels and adjust alignment
plt.yticks(rotation=0, fontsize=14, va='center')  # Adjust y-tick labels font size and vertical alignment
plt.tight_layout(pad=2.0)  # Add padding to layout

# Highlight the diagonal and add scatter points
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j + 0.5, i + 0.5, conf_matrix[i, j],
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='green' if i == j else 'black')

# Adding scatter points to match the visual style
x, y = np.meshgrid(np.arange(conf_matrix.shape[1]), np.arange(conf_matrix.shape[0]))
plt.scatter(x.flatten() + 0.5, y.flatten() + 0.5, s=conf_matrix.flatten() * 10, c='green', alpha=0.6)

plt.show()
