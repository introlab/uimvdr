import os
import pandas as pd
import numpy as np

from scipy import stats


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
    data_path = "/home/jacob/dev/weakseparation/logs/stats/"

    target_class = "speech"
    target_only = False
    metric = "stoi"
    # dataset_id = 3

    ReSpeaker_files = get_files_paths(data_path, target_class, target_only, metric, 0)
    Kinect_files = get_files_paths(data_path, target_class, target_only, metric, 1)
    Sounds16_files = get_files_paths(data_path, target_class, target_only, metric, 2)
    dataset_list = [ReSpeaker_files, Kinect_files, Sounds16_files]

    dataframe_list = []
    for dataset in dataset_list:
        df_list = []
        for file in dataset:
            method = file.split("/")[-2]
            beam = file.split("/")[-1].split("_")[0]
            temp_df = pd.read_csv(file, header=0, names=[method+beam])

            df_list.append(temp_df)

        df = pd.concat(df_list, axis=1)

        dataframe_list.append(df)

    combined_dict = {}
    for df in dataframe_list:
        for col in df.columns:
            if col in combined_dict:
                combined_dict[col] = pd.concat([combined_dict[col], df[col]])
            else:
                combined_dict[col] = df[col]

    combined_df = pd.DataFrame(combined_dict)

    pvalues = pd.DataFrame()
    for beam_col in combined_df.columns:
        if beam_col[-4:] == "Beam":
            si_sdr_col = beam_col[:-4] + metric
            res = stats.ttest_rel(df[beam_col].to_numpy(), df[si_sdr_col].to_numpy(), alternative="greater")
            pvalues[beam_col[:-4]] = res

    print(f"P value: {pvalues}")

if __name__ == "__main__":
    main()

