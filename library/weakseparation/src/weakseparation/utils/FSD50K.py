import os
import json
import csv
import torchaudio
import torch
import random
import h5py

from .Windows import sqrt_hann_window
# from Windows import sqrt_hann_window
from torch.utils.data import Dataset
from scipy import signal


class FSD50KDataset(Dataset):
    def __init__(self, dir, frame_size, hop_size, target_class="Bark", type="train", sample_rate=16000, max_sources=3, forceCPU=False, return_spectrogram=True, rir=True) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.target_class = target_class
        self.gain = 1
        self.type = type
        self.rir = rir
        self.return_spectrogram = return_spectrogram
        ontology_dict = {}
        ontology = json.load(open(os.path.join(dir, "FSD50K.ground_truth", "ontology.json")))
        for item in ontology:
            ontology_dict[item["id"]] = item
        name_to_id = {v['name']: k for k, v in ontology_dict.items()}

        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        # Only use data with one leaf class?
        self.paths_to_data = []
        self.paths_to_target_data = []
        self.labels = {}
        with open(os.path.join(dir, "FSD50K.ground_truth", "dev.csv"), mode='r') as csv_file:
            csvreader = csv.reader(csv_file)
            for idx, row in enumerate(csvreader):
                if row[3] == self.type:
                    ids = row[2].split(",")
                    multiple_leaf_class = False
                    one_leaf_class = False

                    
                    for identifier in ids:
                        if ontology_dict[identifier] and not ontology_dict[identifier]['child_ids']:
                            if one_leaf_class:
                                multiple_leaf_class = True
                            if not one_leaf_class:
                                leaf_class = ontology_dict[identifier]['name']
                                one_leaf_class = True

                    if (one_leaf_class and not multiple_leaf_class) or \
                       (not one_leaf_class and not multiple_leaf_class):
                        if name_to_id[target_class] in ids:
                            self.paths_to_target_data.append(row[0]) 
                        else:
                            self.paths_to_data.append(row[0])
                        list_of_classes = [ontology_dict[n]['name'] for n in ids]
                        self.labels[row[0]] = ' '.join(map(str, list_of_classes))

        if self.rir:
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
        return len(self.paths_to_target_data)
    
    def __getitem__(self, idx):
        wav_path = os.path.join(self.dir, "FSD50K.dev_audio" ,self.paths_to_target_data[idx] + ".wav") 

        x, file_sample_rate = torchaudio.load(wav_path)
        x = torchaudio.functional.resample(x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
        # TODO: number of seconds should be a parameter
        x = self.get_right_number_of_samples(x, self.sample_rate, 5, shuffle=True)

        if self.rir:
            room = random.randint(0, len(self.rooms)-1)
            array_idx = random.randint(0, 5) * 5 
            source_list = []
            rir = self.select_rir(source_list, room, array_idx)
            x = self.apply_rir(rir, x[0])[None,0,:]

        if torch.sum(torch.isnan(x)) >= 1:
            print(wav_path)

        mix = self.stft(x)

        mix = self.normalize(mix, True)

        if not self.return_spectrogram:
            mix = self.istft(mix)

        isolated_sources = mix.clone()[None, ...]

        additionnal_idxs = []
        idxs_classes = [self.target_class]
        for _ in range(self.max_sources-1):
            # TODO: set the probability to 0.5
            if random.random() >= 0:
                while True:
                    additionnal_idx = random.randint(0, len(self.paths_to_data)-1)
                    additionnal_idx_class = self.labels[self.paths_to_data[additionnal_idx]]
                    if additionnal_idx_class not in idxs_classes:
                        additionnal_idxs.append(additionnal_idx)
                        if len(idxs_classes) > 1:
                            idxs_classes[1] += " : " + additionnal_idx_class
                        else:
                            idxs_classes.append(additionnal_idx_class)
                        break

        additional_mix = torch.zeros_like(mix)
        for index in additionnal_idxs:
            wav_path = os.path.join(self.dir, "FSD50K.dev_audio" ,self.paths_to_data[index] + ".wav") 
            additionnal_x, file_sample_rate = torchaudio.load(wav_path)
            additionnal_x = torchaudio.functional.resample(additionnal_x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
            # TODO: number of seconds should be a parameter
            additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, 5, shuffle=True)

            if self.rir:
                rir = self.select_rir(source_list, room, array_idx)
                additionnal_x = self.apply_rir(rir, additionnal_x[0])[None,0,:]

            additionnal_X = self.stft(additionnal_x)
            additionnal_X = self.normalize(additionnal_X, True)
            if not self.return_spectrogram:
                additionnal_X = self.istft(additionnal_X)
                if torch.sum(torch.isnan(additionnal_X)) >= 1:
                    print(wav_path)
                    additionnal_X = torch.zeros_like(additionnal_X)
            
            additional_mix += additionnal_X
            mix += additionnal_X

        isolated_sources = torch.cat((isolated_sources, additional_mix[None, ...]))


        return mix, isolated_sources, idxs_classes
    
    @staticmethod
    def get_right_number_of_samples(x, sample_rate, seconds, shuffle=False):
        nb_of_samples = seconds*sample_rate
        if x.shape[1] < nb_of_samples:
            missing_nb_of_samples = nb_of_samples-x.shape[1]
            random_number = random.randint(0, missing_nb_of_samples)
            x = torch.nn.functional.pad(x, (random_number, missing_nb_of_samples-random_number), mode="constant", value=0)
        elif x.shape[1] > seconds*sample_rate:
            if shuffle:
                random_number = random.randint(0, x.shape[-1]-nb_of_samples-1)
                x = x[..., random_number:nb_of_samples+random_number]
            else:    
                x = x[..., :nb_of_samples]

        return x
    
    @staticmethod
    def apply_rir(rirs, source):
        """
        Method to apply multichannel RIRs to a mono-signal.
        Args:
            rirs (tensor): Multi-channel RIR,   shape = (channels, frames)
            source (tensor): Mono-channel input signal to be reflected to multiple microphones (frames,)
        """
        channels = rirs.shape[0]
        frames = len(source)
        output = torch.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = torch.tensor(
                signal.convolve(source.cpu().numpy(), rirs[channel_index].cpu().numpy())[:frames]
            )

        return output
    
    def select_rir(self, source_list, room, array_idx):
        while (rir_source := random.randint(0, len(self.sources)-1)) in source_list:
            pass

        source_list.append(rir_source)

        rir = self.rir_dataset['rir'][self.rooms[room]][self.sources[rir_source]][()][:,array_idx:array_idx+5]
        rir = rir.transpose()
        rir = torch.tensor(rir).to(torch.float32)
        rir = torchaudio.functional.resample(rir, orig_freq=48000, new_freq=self.sample_rate).to(self.device)
    
        return rir

    
    @staticmethod
    def normalize(X, augmentation = False):
        # Equation: 10*torch.log10((torch.abs(X)**2).mean()) = 0

        if augmentation:
            # Gain between -5 and 5 dB
            aug = torch.rand(1).item()*10 - 5
            augmentation_gain = 10 ** (aug/20)
        else:
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/(torch.abs(X)**2).mean()) 
       
        return augmentation_gain * normalize_gain * X


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    frame_size = 512
    hop_size = int(frame_size / 2)
    target_class = "Speech"
    dataset = FSD50KDataset("/home/jacob/dev/weakseparation/library/dataset/FSD50K", frame_size, hop_size, target_class, forceCPU=True, return_spectrogram=False)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    for _ in range(100):
        for mix, isolatedSources, labels in dataloader:
            pass