import os
import json

def create_labels_json(dataset_dir):
    labels = sorted(os.listdir(dataset_dir))
    data = {'classes': labels}  # Create a dictionary with a key "classes"
    with open('../labels.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Dump the dictionary to JSON

dataset_dir = '../dataset/samples/train'
create_labels_json(dataset_dir)
