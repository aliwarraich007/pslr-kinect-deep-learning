from create_landmarks_single import process_data
import json
trainDir = "../dataset/samples/augmented"
testDir = '../dataset/samples/test'
training_data, validation_data, classes, classes2 = process_data(trainDir, testDir)
traintJson = json.dumps(training_data.tolist())
trainLabelsJson = json.dumps(classes.tolist())
testJson = json.dumps(validation_data.tolist())
testLabelsJson = json.dumps(classes2.tolist())


with open('../landmarks-skeleton-augmented/train.json', 'w') as data_file:
    data_file.write(traintJson)
with open('../labels-skeleton-augmented/trainLabels.json', 'w') as classes_file:
    classes_file.write(trainLabelsJson)
with open('../landmarks-skeleton-augmented/test.json', 'w') as data_file:
    data_file.write(testJson)
with open('../labels-skeleton-augmented/testLabels.json', 'w') as classes_file:
    classes_file.write(testLabelsJson)
