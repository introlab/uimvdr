import os
import pandas as pd
import numpy as np

from scipy import stats


NORMALIZE = True

def get_files_paths(directory, target_class, target_only, metric, dataset_id):
    paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if target_class in root and \
               ((target_only and "Target_only" in root) or \
                (not target_only and not "Target_only" in root)):
                if metric in filename and f"datasetidx_{dataset_id}" in filename:
                    paths.append(os.path.join(root, filename))
    return paths

def main():
    # Create a dictionary
    # data = {'1 sec': [10, 6, 11, 22, 16, 15, 1, 12, 9, 8],
    #         '2 sec': [13, 8, 14, 23, 18, 17, 1, 15, 12, 9],
    #         '5 sec': [13, 8, 14, 25, 20, 17, 4, 17, 12, 12]}
    
    # df = pd.DataFrame(data)

    data_path = "/home/jacob/dev/weakseparation/logs/stats/"

    target_class = "bark"
    target_only = False
    metric = "SI-SDRI"
    dataset_id = 3

    # for root, _, files in os.walk(data_path):
    #     for filename in files:
    #         modified_string = filename.replace(":", "_")
    #         os.rename(os.path.join(root, filename), os.path.join(root, modified_string))

    files = get_files_paths(data_path, target_class, target_only, metric, dataset_id)

    df_list = []
    for file in files:
        method = file.split("/")[-2]
        beam = file.split("/")[-1].split("_")[0]
        temp_df = pd.read_csv(file, header=0, names=[method+beam])

        df_list.append(temp_df)

    df = pd.concat(df_list, axis=1)

    array = df.to_numpy()

    nb_subjects = array.shape[0]
    nb_methods = array.shape[1]

    interaction_degrees_of_freedom = (nb_subjects-1)*(nb_methods-1)

    #Student, p<0.05, 2-tail
    t_factor = stats.t.ppf(1-0.025, interaction_degrees_of_freedom)
    print(f"T: {t_factor}")


    mean_methods = np.mean(array, axis=0)
    mean_subjects = np.mean(array, axis=1)
    global_mean = np.mean(array)

    if NORMALIZE == True:
        normalized_array = array - np.tile(mean_subjects, (nb_methods,1)).T + global_mean

        within_groups_var = np.sum((normalized_array - mean_methods)**2)

        interaction_variance = within_groups_var
    else:
        within_groups_var = np.sum((array - mean_methods)**2)

        between_subjects_var = np.sum((mean_subjects - global_mean)**2) * nb_methods

        interaction_variance = within_groups_var - between_subjects_var


    mean_sum_of_squares_interaction =  interaction_variance / interaction_degrees_of_freedom

    confidence_interval = np.sqrt(mean_sum_of_squares_interaction / nb_subjects) * t_factor

    # Print
    print(df)
    print(mean_methods)
    print(confidence_interval)

if __name__ == "__main__":
    main()

