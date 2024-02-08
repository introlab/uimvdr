import torchaudio
import os

data_folder = "/home/jacob/dev/weakseparation/library/dataset/unsupervised_offline_dataset/SPC2019/flight_task/audio"
separated_folder = "/home/jacob/dev/weakseparation/library/dataset/unsupervised_offline_dataset/SPC2019_team_contributions_separated"
sample_rate_wanted = 16000
nb_of_seconds = 10

def main():
    name_increment = 0
    for filename in os.listdir(data_folder):
    # Check if the current item is a file
        if os.path.isfile(os.path.join(data_folder, filename)):
            # Process the file (print its name for example)
            print(filename[:-4])
            file_path = os.path.join(data_folder, filename)
            waveform, file_sample_rate =  torchaudio.load(file_path)
            waveform = torchaudio.functional.resample(waveform, orig_freq=file_sample_rate, new_freq=sample_rate_wanted)

            total_nb_of_samples = waveform.shape[-1]
            current_sample = 0
            samples_in_recording = nb_of_seconds * sample_rate_wanted

            while current_sample < total_nb_of_samples:
                audio = waveform[..., current_sample:current_sample+samples_in_recording]

                if current_sample + 2*samples_in_recording >= total_nb_of_samples:
                    break

                current_sample += samples_in_recording

                new_file_name = os.path.join(separated_folder, f"{filename[:-4]}_{str(name_increment)}" + ".wav")
                torchaudio.save(new_file_name, audio, sample_rate_wanted)

                name_increment += 1

            audio = waveform[..., current_sample:-1]
            new_file_name = os.path.join(separated_folder, f"{filename[:-4]}_{str(name_increment)}" + ".wav")
            torchaudio.save(new_file_name, audio, sample_rate_wanted)

            current_sample = 0
            name_increment = 0

if __name__ == '__main__':
    main()