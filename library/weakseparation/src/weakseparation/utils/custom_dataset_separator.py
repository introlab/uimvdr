import torchaudio
import os

data_folder = "/home/jacob/dev/weakseparation/library/dataset/Custom/home/respeaker"
separated_folder = "/home/jacob/dev/weakseparation/library/dataset/Custom/home_separated/respeaker"
nb_of_seconds = 5
nb_of_pause = 2
respeaker_delay = 1
kinect_delay = 1.366
sounds_delay = 1.770
nb_of_files = 36

def main():
    name_increment = 0
    for (dirpath, dirnames, _) in os.walk(data_folder):
        for dir in dirnames:
            for (_, _, filenames) in os.walk(os.path.join(dirpath, dir)):
                for f in filenames:
                    file_path = os.path.join(dirpath, dir, f)
                    waveform, sample_rate =  torchaudio.load(file_path)

                    current_sample = int(respeaker_delay*sample_rate)
                    samples_in_recording = nb_of_seconds*sample_rate
                    samples_in_between = nb_of_pause*sample_rate

                    while name_increment < nb_of_files:
                        if f[:-4] == 'chirp':
                            audio = waveform[1:, current_sample:current_sample+(60*sample_rate)]
                        else:
                            audio = waveform[1:, current_sample:current_sample+samples_in_recording]


                        current_sample += samples_in_recording+samples_in_between
                        dir_path = os.path.join(separated_folder, dir, f[:-4])
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)

                        file_name = os.path.join(dir_path, str(name_increment) + ".wav")
                        torchaudio.save(file_name, audio, sample_rate)

                        if f[:-4] == 'chirp':
                            break

                        name_increment += 1

                    current_sample = 0
                    name_increment = 0

if __name__ == '__main__':
    main()