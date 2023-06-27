import os
import torch
import torchaudio
import h5py
import numpy as np
import random
import math

from scipy import signal
from torch.utils.data import Dataset
# from .Windows import sqrt_hann_window

def sqrt_hann_window(
    window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False
):
    return torch.sqrt(
        torch.hann_window(
            window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )
    )


class FUSSDataset(Dataset):
    def __init__(self, dir, frame_size, hop_size, type="train", sample_rate=16000, max_sources=4, forceCPU=False, return_spectrogram=True) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.max_sources = max_sources
        self.gain = 1
        self.type = type
        self.return_spectrogram = return_spectrogram

        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        if type == "train":
            self.split = os.path.join(self.dir, "train_example_list.txt")
        elif type == "val":
            self.split = os.path.join(self.dir, "validation_example_list.txt")
        elif type == "predict":
            self.split = os.path.join(self.dir, "eval_example_list.txt")
        else:
            raise RuntimeError("Unknown dataset type")

        self.paths_to_mix = []
        self.paths_to_isolated = []
        with open(self.split, mode='r') as file:
            examples = file.readlines()
            for example in examples:
                data_paths = example.replace("\n", "").split("\t")
                if len(data_paths[1:]) <= max_sources:
                # if len(data_paths[1:]) == max_sources:
                    self.paths_to_mix.append(data_paths[0])
                    self.paths_to_isolated.append(data_paths[1:])

        # RIRS
        self.path_to_rirs = "/home/jacob/dev/weakseparation/library/dataset/dEchorate/dEchorate_rir.h5"
        self.rir_dataset = h5py.File(self.path_to_rirs, mode='r')
        self.rooms = list(self.rir_dataset['rir'].keys())
        self.sources = list(self.rir_dataset['rir'][self.rooms[0]].keys())

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
        return len(self.paths_to_mix)
    
    def __getitem__(self, idx):
        #Nb of sources, 5 mics, 10 sec at 16
        if self.return_spectrogram:
            isolated_sources = torch.zeros(
                self.max_sources, 
                5, # Nb of mics
                self.frame_size//2 +1, #Nb of bins
                int(10*self.sample_rate/self.hop_size) + 1, #Nb of frames
                dtype=torch.complex64
            )
        else:
            isolated_sources = torch.zeros(self.max_sources, 5, 10*self.sample_rate)
        room = random.randint(0, len(self.rooms)-1)
        array_idx = random.randint(0, 5) * 5 
        source_list = []
        for source_idx, source in enumerate(self.paths_to_isolated[idx]):
            isolated_src, file_sample_rate = torchaudio.load(os.path.join(self.dir, source))
            isolated_src = torchaudio.functional.resample(isolated_src, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
            isolated_src = self.get_right_number_of_samples(isolated_src, 10)

            while (rir_source := random.randint(0, len(self.sources)-1)) in source_list:
                pass

            source_list.append(rir_source)

            rir = self.rir_dataset['rir'][self.rooms[room]][self.sources[rir_source]][()][:,array_idx:array_idx+5]
            rir = rir.transpose()
            rir = torch.tensor(rir).to(torch.float32)
            rir = torchaudio.functional.resample(rir, orig_freq=48000, new_freq=self.sample_rate).to(self.device)

            # Only apply rir to foreground sounds 
            if source_idx != 0:
                isolated_src = self.apply_rir(rir, isolated_src[0])

            isolated_src = self.stft(isolated_src)

            # Does this work in time domain?
            isolated_src = self.normalize(isolated_src, True)

            if not self.return_spectrogram:
                isolated_src = self.istft(isolated_src)

            isolated_sources[source_idx] = isolated_src

        mix = torch.sum(isolated_sources, dim=0, keepdim=False)
        
        return mix*self.gain, isolated_sources*self.gain, torch.tensor([])
    
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
        # Equation: 10*torch.log10((torch.abs(X)**2).mean()) = 0

        if augmentation:
            aug = torch.rand(1).item()*10 - 5
            augmentation_gain = 10 ** (aug/20)
        else:
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/(torch.abs(X)**2).mean()) 
       
        return augmentation_gain * normalize_gain * X

    @staticmethod
    def apply_rir(rirs, source):
        """
        Method to apply multichannel RIRs to a mono-signal.
        Args:
            rirs (ndarray): Multi-channel RIR,   shape = (channels, frames)
            source (ndarray): Mono-channel input signal to be reflected to multiple microphones (frames,)
        """
        channels = rirs.shape[0]
        frames = len(source)
        output = torch.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = torch.tensor(
                signal.convolve(source.cpu().numpy(), rirs[channel_index].cpu().numpy())[:frames]
            )

        return output

if __name__ == "__main__":
    random.seed(10)
    dataset = FUSSDataset("/home/jacob/dev/weakseparation/library/dataset/FUSS/FUSS_ssdata/ssdata", 512, 256, forceCPU=True, return_spectrogram=False)
    iterable = iter(dataset)
    next(iterable)
    mix, isolated, _ = next(iterable)

    torchaudio.save("/home/jacob/dev/mix.wav", mix, 16000)

    for idx,src in enumerate(isolated):
        torchaudio.save(f"/home/jacob/dev/source{idx}.wav", src, 16000)

