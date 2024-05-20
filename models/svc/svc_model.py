import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, learning_curve, cross_val_score
from scipy.stats import uniform
from joblib import dump

# Load training data
json_file_path_train_data = "../../landmarks/landmarks_augmented/train.json"
json_file_path_train_labels = "../../landmarks/labels_augmented/trainLabels.json"

with open(json_file_path_train_data, 'r') as json_file:
    training_data = json.load(json_file)

with open(json_file_path_train_labels, 'r') as json_file:
    training_labels = json.load(json_file)

training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Load testing data
json_file_path_test_data = "../../landmarks/landmarks_augmented/test.json"
json_file_path_test_labels = "../../landmarks/labels_augmented/testLabels.json"

with open(json_file_path_test_data, 'r') as json_file:
    testing_data = json.load(json_file)

with open(json_file_path_test_labels, 'r') as json_file:
    testing_labels = json.load(json_file)

testing_data = np.array(testing_data)
testing_labels = np.array(testing_labels)


# Placeholder for data augmentation
# Add your data augmentation code here if you have any.

# Feature Engineering
def extract_features(data):
    # Assuming data shape is (n_samples, n_timesteps, n_features)
    velocities = np.diff(data, axis=1)
    accelerations = np.diff(velocities, axis=1)

    # Flatten and concatenate original data with velocities and accelerations
    flattened_data = data.reshape(data.shape[0], -1)
    flattened_velocities = velocities.reshape(velocities.shape[0], -1)
    flattened_accelerations = accelerations.reshape(accelerations.shape[0], -1)

    return np.hstack((flattened_data, flattened_velocities, flattened_accelerations))


training_data_features = extract_features(training_data)
testing_data_features = extract_features(testing_data)

# Feature scaling
scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_data_features)
testing_data_scaled = scaler.transform(testing_data_features)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'C': uniform(loc=0, scale=4),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist,
                                   n_iter=100, cv=5, n_jobs=-1, verbose=2,
                                   random_state=42)
random_search.fit(training_data_scaled, training_labels)

# Best model
best_svc_model = random_search.best_estimator_

# Predict labels for training and testing data
training_predicted_labels_svc = best_svc_model.predict(training_data_scaled)
testing_predicted_labels_svc = best_svc_model.predict(testing_data_scaled)

# Calculate accuracy of training and testing data
training_accuracy_svc = accuracy_score(training_labels, training_predicted_labels_svc)
testing_accuracy_svc = accuracy_score(testing_labels, testing_predicted_labels_svc)
print("Training Accuracy (SVC):", training_accuracy_svc)
print("Testing Accuracy (SVC):", testing_accuracy_svc)

# Print classification report and confusion matrix for detailed evaluation
print("\nClassification Report for Testing Data (SVC):")
print(classification_report(testing_labels, testing_predicted_labels_svc))

print("\nConfusion Matrix for Testing Data (SVC):")
conf_matrix_svc = confusion_matrix(testing_labels, testing_predicted_labels_svc)
print(conf_matrix_svc)

# Save the best SVC model and scaler
with open("./psl_augmented.pkl", 'wb') as model_file:
    pickle.dump(best_svc_model, model_file)

dump(scaler, './scaler_augmented.joblib')

# Plot confusion matrix using Seaborn for SVC
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svc, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(testing_labels),
            yticklabels=np.unique(testing_labels))
plt.title('Confusion Matrix (SVC)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Train and evaluate Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(training_data_scaled, training_labels)

# Predict labels for training and testing data
training_predicted_labels_rf = rf_model.predict(training_data_scaled)
testing_predicted_labels_rf = rf_model.predict(testing_data_scaled)

# Calculate accuracy of training and testing data
training_accuracy_rf = accuracy_score(training_labels, training_predicted_labels_rf)
testing_accuracy_rf = accuracy_score(testing_labels, testing_predicted_labels_rf)
print("Training Accuracy (RF):", training_accuracy_rf)
print("Testing Accuracy (RF):", testing_accuracy_rf)

# Print classification report and confusion matrix for detailed evaluation
print("\nClassification Report for Testing Data (RF):")
print(classification_report(testing_labels, testing_predicted_labels_rf))

print("\nConfusion Matrix for Testing Data (RF):")
conf_matrix_rf = confusion_matrix(testing_labels, testing_predicted_labels_rf)
print(conf_matrix_rf)

# Save the Random Forest model
with open("./rf_psl_augmented.pkl", 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Plot confusion matrix using Seaborn for Random Forest
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(testing_labels),
            yticklabels=np.unique(testing_labels))
plt.title('Confusion Matrix (RF)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, training_data_scaled, training_labels, cv=5)
print("Cross-validation scores (RF):", cv_scores_rf)
print("Mean cross-validation score (RF):", np.mean(cv_scores_rf))

# Plot learning curves for the best SVC model
train_sizes, train_scores, test_scores = learning_curve(best_svc_model, training_data_scaled, training_labels, cv=5,
                                                        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curves (SVM)")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.grid()

# Plot learning curve
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()
