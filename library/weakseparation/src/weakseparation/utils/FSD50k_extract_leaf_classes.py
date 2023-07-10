import json
import csv
import os
import torchaudio

WRITE = True

fsd50k_eval_path = "/home/jacob/Dev/weakseparation/library/dataset/FSD50K/FSD50K.ground_truth/eval.csv"
fsd50k_writing_path = "/home/jacob/Dev/weakseparation/library/dataset/FSD50K/FSD50K.ground_truth/piano.csv"

ontology_path = "/home/jacob/Dev/weakseparation/library/dataset/FSD50K/ontology.json"
ontology_file = open(ontology_path)
  
ontology = json.load(ontology_file)
ontology_dict = {}
for item in ontology:
    ontology_dict[item["id"]] = item

files_with_only_one_leaf_class = []

with open(fsd50k_eval_path, mode='r') as csv_file:
    csvreader = csv.reader(csv_file)

    # Skip header
    next(csvreader, None)

    # Process labels
    for idx, row in enumerate(csvreader):
        ids = row[2].split(",")
        leaf_class = False
        writing_row = {}

        for identifier in ids:
            # if ontology_dict[identifier] and not ontology_dict[identifier]['child_ids']:
            if ontology_dict[identifier] and ontology_dict[identifier]['name'] == "Piano":
                writing_row = {
                    'fname' : row[0],
                    'class' : ontology_dict[identifier]['name'],
                    'id' : identifier,
                    }

                # if leaf_class:
                #     writing_row = {}

                # leaf_class = True

        if writing_row:
            files_with_only_one_leaf_class.append(writing_row)

print(f'There is {len(files_with_only_one_leaf_class)} files with only one leaf class')

# Compute length of audio for a class
audio_path = "/home/jacob/Dev/weakseparation/library/dataset/FSD50K/FSD50K.eval_audio"
class_name = "Piano"
nb_of_seconds = 0
nb_of_files = 0
for file in files_with_only_one_leaf_class:
    if file['class'] == class_name:
        x, file_sample_rate = torchaudio.load(os.path.join(audio_path, file["fname"] + ".wav"))
        samples = x.shape[1]
        nb_of_seconds += samples/file_sample_rate
        nb_of_files += 1


print(f'{class_name}: {nb_of_seconds:.2f} sec in {nb_of_files} files')


if WRITE:
    with open(fsd50k_writing_path, mode='w') as csv_file:
        fieldnames = ['fname', 'class', 'id']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for file in files_with_only_one_leaf_class:
            if file:
                writer.writerow(file)
print("end")