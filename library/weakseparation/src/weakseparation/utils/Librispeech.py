import os
import re
import csv
import torchaudio
import torch

from .LabelUtils import class_to_id
from .Windows import sqrt_hann_window
from torch.utils.data import Dataset


class LibrispeechDataset(Dataset):
    def __init__(self, dir, frame_size, hop_size, type="train", sample_rate=16000, max_sources=3, forceCPU=False, return_spectrogram=True) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.gain = 1
        self.type = type
        self.return_spectrogram =return_spectrogram

        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        if type == "train":
            root = os.path.join(self.dir, "train")
        elif type == "val":
            # root = os.path.join(self.dir, "train")
            root = os.path.join(self.dir, "val")
        else:
            raise RuntimeError("Unknown dataset type")
        self.type = "train"

        self.paths_to_data = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name[-5:] == ".flac":
                    filename = os.path.join(path, name)
                    self.paths_to_data.append(filename)

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=frame_size,
            hop_length=hop_size,
            power=None,
            window_fn=sqrt_hann_window
        )

        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=frame_size, hop_length=hop_size, window_fn=sqrt_hann_window
        )


    def __len__(self):
        return len(self.paths_to_data)

    def __getitem__(self, idx):
        wav_path = self.paths_to_data[idx]

        # Get label
        speaker_ids = [self.get_speaker_id(wav_path)]
        sample_labels = torch.tensor([class_to_id["speaker"]])

        x, file_sample_rate = torchaudio.load(wav_path)
        x = x.to(self.device)

        x = torchaudio.functional.resample(x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)

        # TODO: change number of seconds?
        mix = self.get_right_number_of_samples(x, 3, True if self.type == "train" else False)
        # mix = self.get_right_number_of_samples(x, 3, False)
        
        mix = self.stft(mix)

        mix = self.normalize(mix, True if self.type == "train" else False)
        # mix = self.normalize(mix, False)
        if not self.return_spectrogram:
            mix = self.istft(mix)

        isolated_sources = mix[None, ...]

        if self.type == "train":
            additionnal_idxs = []
            for _ in range(self.max_sources-1):
                # TODO: set the probability to 0.5
                if torch.rand(1).item() <= 1:
                    while True:
                        additionnal_idx = torch.randint(low=0, high=len(self)-1, size=(1,))[0].item()
                        speaker_id = self.get_speaker_id(self.paths_to_data[additionnal_idx])
                        
                        if additionnal_idx != idx and not additionnal_idx in additionnal_idxs and not speaker_id in speaker_ids:
                            break
                    additionnal_idxs.append(additionnal_idx)
                    speaker_ids.append(speaker_id)

            for idx in additionnal_idxs:
                wav_path = self.paths_to_data[idx]
                additionnal_x, _ = torchaudio.load(wav_path)
                additionnal_x = additionnal_x.to(self.device)
                additionnal_x = torchaudio.functional.resample(
                    additionnal_x,
                    orig_freq=file_sample_rate,
                    new_freq=self.sample_rate
                ).to(self.device)
                additionnal_x = self.get_right_number_of_samples(additionnal_x, 3, True)
                additionnal_x = self.stft(additionnal_x)
                additionnal_x = self.normalize(additionnal_x, True)
                if not self.return_spectrogram:
                    additionnal_x = self.istft(additionnal_x)
                isolated_sources = torch.cat((isolated_sources, additionnal_x[None, ...]))
                sample_labels = torch.cat(
                    (sample_labels,
                    torch.tensor([class_to_id["speaker"]])))
                mix += additionnal_x

            while isolated_sources.shape[0] < self.max_sources:
                zeroes = torch.zeros_like(mix)
                isolated_sources = torch.cat((isolated_sources, zeroes[None, ...]))
                sample_labels = torch.cat((
                    sample_labels,
                    torch.tensor([class_to_id["nothing"]])))
        
        return mix*self.gain, isolated_sources*self.gain, sample_labels

    
    @staticmethod
    def get_speaker_id(wav_path):
        labelwavName = wav_path.split("/")[-1]
        speaker_id = labelwavName.split("-")[0]
        return speaker_id

    def get_right_number_of_samples(self, x, seconds, shuffle=False):
        nb_of_samples = seconds*self.sample_rate
        if x.shape[1] < nb_of_samples:
            x = torch.nn.functional.pad(x, (0, nb_of_samples-x.shape[1]), mode="constant", value=0)
        elif x.shape[1] > seconds*self.sample_rate:
            if shuffle:
                random_number = torch.randint(low=0, high=x.shape[-1]-nb_of_samples-1, size=(1,))[0].item()
                x = x[..., random_number:nb_of_samples+random_number]
            else:    
                x = x[..., :nb_of_samples]

        return x

    @staticmethod
    def normalize(X, augmentation = False):
        # Equation: 10*torch.log10((torch.abs(X)**2).max()) = 0, max instead of mean because inital level

        if augmentation:
            aug = torch.rand(1).item()*10 - 5
            augmentation_gain = 10 ** (aug/20)
        else:
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/(torch.abs(X)**2).mean()) 
       
        return augmentation_gain * normalize_gain * X


if __name__ == '__main__':
    dataset = LibrispeechDataset("/home/jacob/dev/weakseparation/library/dataset/Librispeech", 512, 256, forceCPU=True)
    print(len(dataset))
    print(next(iter(dataset)))
