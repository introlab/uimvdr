import torch
import torchaudio
import os

data_folder = "/home/jacob/dev/weakseparation/library/dataset/Custom/2008"
separated_data_folder = "/home/jacob/dev/weakseparation/library/dataset/Custom/2008_separated"

def compute_snr(target, noise):
    energy_target = torch.mean(target**2)
    energy_noise = torch.mean(noise**2)

    snr = 10*torch.log10((energy_target-energy_noise)/energy_noise)

    return snr


def main():
    mean_snr = 0
    nb_of_files = 0
    for (dirpath, dirnames, _) in os.walk(separated_data_folder):
        for dir in dirnames:
            for (sub_dir, _, filenames) in os.walk(os.path.join(dirpath, dir)):
                for f in filenames:
                    file_path = os.path.join(sub_dir, f)

  
                    silence_waveform_path = sub_dir.split("/")
                    position_dir = silence_waveform_path[-1]
                    room_dir = silence_waveform_path.pop(-4)
                    room_dir = room_dir[:-10]
                    silence_waveform_path.insert(-3, room_dir)
                    silence_waveform_path = "/".join(silence_waveform_path)
                    silence_waveform_path = silence_waveform_path + "_silence.wav"

                    if position_dir == 'chirp':
                        pass
                    else:
                        waveform, sample_rate =  torchaudio.load(file_path)
                        silence_waveform, sample_rate =  torchaudio.load(silence_waveform_path)

                        # if torch.sum(torch.isnan(silence_waveform)):
                        #     print(silence_waveform_path)
                        
                        # if torch.sum(torch.isnan(waveform)):
                        #     print(file_path)

                        snr = compute_snr(waveform, silence_waveform)
                        if not (torch.isnan(snr).item()):
                            mean_snr += snr
                            nb_of_files += 1
                        else:
                            print(silence_waveform_path)
                            print(file_path)

    print(f"Mean {mean_snr/nb_of_files}")
    print(f"Number of files: {nb_of_files}")
    print(f"SNR: {mean_snr}")


if __name__ == '__main__':
    main()