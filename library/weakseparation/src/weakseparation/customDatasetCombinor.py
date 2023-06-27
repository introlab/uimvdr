import os
import torchaudio
import torch
import random

data_to_play_folder = "/home/jacob/Dev/weakseparation/library/dataset/data_to_play"
nb_of_seconds = 5

def get_right_number_of_samples(x, sample_rate, seconds, shuffle=False):
    nb_of_samples = seconds*sample_rate
    if x.shape[1] < nb_of_samples:
        missing_nb_of_samples = nb_of_samples-x.shape[1]
        random_number = random.randint(0, missing_nb_of_samples)
        x = torch.nn.functional.pad(x, (random_number, missing_nb_of_samples-random_number), mode="constant", value=0)
    elif x.shape[1] > seconds*sample_rate:
        if shuffle:
            random_number = torch.randint(low=0, high=x.shape[-1]-nb_of_samples-1, size=(1,))[0].item()
            x = x[..., random_number:nb_of_samples+random_number]
        else:    
            x = x[..., :nb_of_samples]

    return x

def main():
    for (dirpath, dirnames, filenames) in os.walk(data_to_play_folder):
        full_waveform = None
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            waveform, sample_rate =  torchaudio.load(file_path)
            # waveform = get_right_number_of_samples(waveform, sample_rate, nb_of_seconds)
            # Pad at the end to ensure no more reverb is present
            nb_of_samples = 2*sample_rate
            waveform = torch.nn.functional.pad(waveform, (0, nb_of_samples), mode="constant", value=0)

            if full_waveform is None:
                full_waveform = waveform
            else:
                full_waveform = torch.cat((full_waveform, waveform), 1)

        new_file = dirpath.split('/')[-1] + ".wav"
        class_file_path = os.path.join(dirpath, new_file)

        if full_waveform is not None:
            torchaudio.save(class_file_path, full_waveform, sample_rate)

if __name__ == '__main__':
    main()