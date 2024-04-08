import os
import random
import shutil


def get_files_paths(directory):
    paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            paths.append(os.path.join(root, filename))

    return paths

# Function to split files into train, validation, and test sets
def split_files(file_paths, train_ratio=0.95, val_ratio=0.025, test_ratio=0.025):
    # Shuffle the list of file paths
    random.shuffle(file_paths)
    
    # Calculate number of files for each set
    total_files = len(file_paths)
    num_train = int(total_files * train_ratio)
    num_val = int(total_files * val_ratio)
    num_test = int(total_files * test_ratio)
    
    # Split the file paths
    train_files = file_paths[:num_train]
    val_files = file_paths[num_train:num_train + num_val]
    test_files = file_paths[num_train + num_val:]
    
    return train_files, val_files, test_files

if __name__ == "__main__":
    # Directory path containing files
    directory = "/home/jacob/dev/weakseparation/library/dataset/drone_dataset/darit_audio"
    train_directory = "/home/jacob/dev/weakseparation/library/dataset/drone_dataset/train"
    val_dir = "/home/jacob/dev/weakseparation/library/dataset/drone_dataset/val"
    test_dir = "/home/jacob/dev/weakseparation/library/dataset/drone_dataset/test"

    
    # Get list of all file paths in the directory
    file_paths = get_files_paths(directory)
    
    # Split files into train, validation, and test sets
    train_files, val_files, test_files = split_files(file_paths)
    
    # Print number of files in each set
    print("Number of files in training set:", len(train_files))
    print("Number of files in validation set:", len(val_files))
    print("Number of files in test set:", len(test_files))

    for file in train_files:
        shutil.copy(file, train_directory)

    for file in val_files:
        shutil.copy(file, val_dir)

    for file in test_files:
        shutil.copy(file, test_dir)
