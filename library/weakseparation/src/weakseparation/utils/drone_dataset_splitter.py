import os
import random
import shutil

from glob import glob


def get_files_paths(directory):
    file_paths = glob(f"{directory}/**/*.wav", recursive=True)

    return file_paths

# Function to split files into train, validation, and test sets
def split_files(file_paths, train_ratio=0.95, val_ratio=0.025, test_ratio=0.025):
    # Shuffle the list of file paths
    random.shuffle(file_paths)
    
    # Calculate number of files for each set
    total_files = len(file_paths)
    num_train = int(total_files * train_ratio)
    train_files = file_paths[:num_train]
    if val_ratio > 0:
        num_val = int(total_files * val_ratio)
        val_files = file_paths[num_train:num_train + num_val]
    else:
        val_files = []
    if test_ratio > 0:
        num_test = int(total_files * test_ratio)
        test_files = file_paths[num_train + num_val:]
    else:
        test_files = []
    
    return train_files, val_files, test_files

if __name__ == "__main__":
    # Directory path containing files
    directory = "/home/jacob/dev/weakseparation/library/dataset/XPRIZE/Screaming-Piha"
    train_directory = "/home/jacob/dev/weakseparation/library/dataset/XPRIZE/bird/train"
    val_dir = "/home/jacob/dev/weakseparation/library/dataset/XPRIZE/bird/val"
    test_dir = "/home/jacob/dev/weakseparation/library/dataset/XPRIZE/bird/test"

    
    # Get list of all file paths in the directory
    file_paths = get_files_paths(directory)
    
    # Split files into train, validation, and test sets
    train_files, val_files, test_files = split_files(file_paths, train_ratio=0.95, val_ratio=0.05, test_ratio=0)
    
    # Print number of files in each set
    print("Number of files in training set:", len(train_files))
    print("Number of files in validation set:", len(val_files))
    print("Number of files in test set:", len(test_files))

    # for file in train_files:
    #     shutil.copy(file, train_directory)

    # for file in val_files:
    #     shutil.copy(file, val_dir)

    # for file in test_files:
    #     shutil.copy(file, test_dir)

    for file in train_files:
        shutil.move(file, train_directory)

    for file in val_files:
        shutil.move(file, val_dir)

    for file in test_files:
        shutil.move(file, test_dir)
