from skeleton_landmarks import process_videos_in_dataset
import json

# Directories for dataset
trainDir = "../dataset/samples/train"
testDir = '../dataset/samples/test'

# Process and save training data
train_data, train_classes = process_videos_in_dataset(trainDir, '../landmarks-skeleton-augmented/train')
trainJson = json.dumps(train_data)
trainLabelsJson = json.dumps(train_classes)

with open('../landmarks-skeleton-augmented/train.json', 'w') as data_file:
    data_file.write(trainJson)
with open('../landmarks-skeleton-augmented/trainLabels.json', 'w') as classes_file:
    classes_file.write(trainLabelsJson)

# Process and save testing data
test_data, test_classes = process_videos_in_dataset(testDir, '../landmarks-skeleton-augmented/test')
testJson = json.dumps(test_data)
testLabelsJson = json.dumps(test_classes)

with open('../landmarks-skeleton-augmented/test.json', 'w') as data_file:
    data_file.write(testJson)
with open('../landmarks-skeleton-augmented/testLabels.json', 'w') as classes_file:
    classes_file.write(testLabelsJson)
