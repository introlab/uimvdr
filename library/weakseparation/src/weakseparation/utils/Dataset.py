import os
import re
import csv
import torchaudio
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class SeclumonsDataset(Dataset):
    def __init__(self, dir, train=True) -> None:
        super().__init__()
        self.dir = dir
        if train:
            self.split = os.path.join(self.dir, "unilabel_split1_train.csv")
        else:
            self.split = os.path.join(self.dir, "unilabel_split1_test.csv")

        self.paths_to_data = []
        with open(self.split, mode='r') as csv_file:
            csvreader = csv.reader(csv_file)
            for row in csvreader:
                self.paths_to_data.append(row[0])
                
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

        self.transformation = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            hop_length=256,
            power=None,
            return_complex=True,
        )


    def __len__(self):
        return len(self.paths_to_data)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.dir, self.paths_to_data[idx])

        x, sample_rate = torchaudio.load(wav_path)

        X = self.transformation(x)

        # self.plot_spectrogram_from_spectrogram(X)
        # self.plot_spectrogram_from_waveform(x, sample_rate)
        # self.plot_waveform(x, sample_rate)
        
        # Get label
        wavName = wav_path.split("/")[-1]
        wavName = re.sub("_micro[0-9]","", wavName)
        label = self.labels[wavName]
        
        return X, label

    @staticmethod
    def plot_spectrogram_from_spectrogram(spectrogram, title="Spectrogram"):
        plt.figure(title)
        plt.imshow(20*torch.abs(spectrogram).log10()[0,:,:].numpy(), origin='lower')
        plt.colorbar()

    @staticmethod
    def plot_spectrogram_from_waveform(waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show()

    @staticmethod
    def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show()
        


if __name__ == '__main__':
    dataset = SeclumonsDataset("/home/jacob/dev/weakseparation/library/dataset/SECL-UMONS")
    print(len(dataset))
    print(next(iter(dataset)))
