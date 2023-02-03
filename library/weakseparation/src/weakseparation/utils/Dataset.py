import os
import re
import csv
import torchaudio
import torch

from .PlotUtils import *
from .LabelUtils import class_to_id
from .Windows import sqrt_hann_window
from torch.utils.data import Dataset


class SeclumonsDataset(Dataset):
    def __init__(self, dir, frame_size, hop_size, type="train", sample_rate=16000, max_sources=3, forceCPU=False) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.gain = 10

        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        if type == "train":
            self.split = os.path.join(self.dir, "unilabel_split1_train.csv")
            # self.split = os.path.join(self.dir, "unilabel_split1_speech_train.csv")
        elif type == "val":
            self.split = os.path.join(self.dir, "unilabel_split1_test.csv")
            # self.split = os.path.join(self.dir, "unilabel_split1_speech_test.csv")
        elif type == "predict":
            self.split = os.path.join(self.dir, "unilabel_split1_predict.csv")
        else:
            raise RuntimeError("Unknown dataset type")

        self.paths_to_data = []
        self.idx_from_room1_to_room2 = 0
        with open(self.split, mode='r') as csv_file:
            csvreader = csv.reader(csv_file)
            for idx, row in enumerate(csvreader):
                self.paths_to_data.append(row[0])
                if re.search("room1", row[0]):
                    self.idx_from_room1_to_room2 = idx
                
        unilabelDataPath = os.path.join(dir,"Unilabel")

        # All labels (including test and train)
        self.labels = {}
        for subdir, dirs, files in os.walk(unilabelDataPath):
            for file in files:
                if (file[-3:] == "csv"):
                    with open(os.path.join(subdir, file), mode='r') as csv_file:
                        csvreader = csv.DictReader(csv_file)
                        for row in csvreader:
                            self.labels[row["File name"]] = row

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=frame_size,
            hop_length=hop_size,
            power=None,
            window_fn=sqrt_hann_window
        )


    def __len__(self):
        return len(self.paths_to_data)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.dir, self.paths_to_data[idx])

        # Get label
        initial_label = self.labels[self.get_name_for_annotations(wav_path)]
        sample_labels = torch.tensor([int(self.labels[self.get_name_for_annotations(wav_path)]["class number"])], dtype=torch.int)

        x, file_sample_rate = self.get_multi_channel(wav_path)

        x = torchaudio.functional.resample(x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)

        # TODO: change number of seconds
        x = self.get_right_number_of_samples(x, 3)

        mix = self.stft(x)

        isolated_sources = mix[None, ...]

        additionnal_idxs = []
        for _ in range(self.max_sources-1):
            # TODO: set the probability to 0.5
            if torch.rand(1)[0] <= 1:
                while True:
                    if initial_label["room number"] == '1':
                        additionnal_idx = torch.randint(low=0, high=self.idx_from_room1_to_room2+1, size=(1,))[0].item()
                    else:
                        additionnal_idx = torch.randint(low=self.idx_from_room1_to_room2+1, high=len(self)-1, size=(1,))[0].item()
                    
                    if additionnal_idx != idx and not additionnal_idx in additionnal_idxs:
                        break
                additionnal_idxs.append(additionnal_idx)

        for idx in additionnal_idxs:
            wav_path = os.path.join(self.dir, self.paths_to_data[idx])
            additionnal_x, _ = self.get_multi_channel(wav_path)
            additionnal_x = torchaudio.functional.resample(
                additionnal_x,
                orig_freq=file_sample_rate,
                new_freq=self.sample_rate
            ).to(self.device)
            additionnal_x = self.get_right_number_of_samples(additionnal_x, 3)
            additionnal_X = self.stft(additionnal_x)
            isolated_sources = torch.cat((isolated_sources, additionnal_X[None, ...]))
            sample_labels = torch.cat(
                (sample_labels,
                torch.tensor([int(self.labels[self.get_name_for_annotations(wav_path)]["class number"])])))
            mix += additionnal_X

        while isolated_sources.shape[0] < self.max_sources:
            zeroes = torch.zeros_like(mix)
            isolated_sources = torch.cat((isolated_sources, zeroes[None, ...]))
            sample_labels = torch.cat((
                sample_labels,
                torch.tensor([class_to_id["nothing"]])))


        # plot_spectrogram_from_spectrogram(mix.cpu())
        # plot_spectrogram_from_waveform(x.cpu(), self.sample_rate)
        # plot_waveform(x.cpu(), sample_rate)

        # istft = torchaudio.transforms.InverseSpectrogram(
        #         n_fft=512, hop_length=256, window_fn=sqrt_hann_window
        #     )
        # resampled_x = istft(mix)
        # torchaudio.save(f'./test.wav', resampled_x.cpu(), self.sample_rate)
        
        return mix, isolated_sources, sample_labels


    @staticmethod
    def get_name_for_annotations(wav_path):
        labelwavName = wav_path.split("/")[-1]
        return re.sub("_micro[0-9]","", labelwavName)

    def get_right_number_of_samples(self, x, seconds):
        if x.shape[1] < seconds*self.sample_rate:
            x = torch.nn.functional.pad(x, (0, seconds*self.sample_rate-x.shape[1]), mode="constant", value=0)
        elif x.shape[1] > seconds*self.sample_rate:
            x = x[..., :3*self.sample_rate-x.shape[1]]

        return x

    def get_multi_channel(self, wav_path):
        x = None
        for mic in range(7):
            wav_mic_path = re.sub("micro[0-9]",f"micro{mic}", wav_path)

            single_mic, file_sample_rate = torchaudio.load(wav_mic_path)

            if x is None:
                x = single_mic
            else:
                x = torch.cat((x, single_mic), 0)
        return x*self.gain, file_sample_rate
        


if __name__ == '__main__':
    dataset = SeclumonsDataset("/home/jacob/Dev/weakseparation/library/dataset/SECL-UMONS")
    print(len(dataset))
    print(next(iter(dataset)))
