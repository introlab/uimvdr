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

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['display_name']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

class FSD50KDataset(Dataset):
    def __init__(self, 
                 dir,
                 frame_size, 
                 hop_size, 
                 target_class="Bark",
                 non_mixing_classes = ["Dog"],
                 type="train", 
                 sample_rate=16000, 
                 max_sources=3, 
                 forceCPU=False, 
                 return_spectrogram=True, 
                 rir=False,
                 supervised=True,
                 isolated=False) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.target_class = target_class
        self.nb_of_seconds = 3
        self.non_mixing_classes = non_mixing_classes.copy()
        self.non_mixing_classes.append(target_class)
        self.type = type
        self.rir = rir
        self.return_spectrogram = return_spectrogram
        self.supervised = supervised
        self.isolated = isolated
        self.ontology_dict = {}
        ontology = json.load(open(os.path.join(dir, "FSD50K.ground_truth", "ontology.json")))
        for item in ontology:
            self.ontology_dict[item["id"]] = item
        name_to_id = {v['name']: k for k, v in self.ontology_dict.items()}

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
                    if isolated:
                        multiple_leaf_class = False
                        one_leaf_class = False
                        
                        for identifier in ids:
                            if self.ontology_dict[identifier] and not self.ontology_dict[identifier]['child_ids']:
                                if one_leaf_class:
                                    multiple_leaf_class = True
                                if not one_leaf_class:
                                    leaf_class = self.ontology_dict[identifier]['name']
                                    one_leaf_class = True

                        if (one_leaf_class and not multiple_leaf_class) or \
                        (not one_leaf_class and not multiple_leaf_class):
                            if name_to_id[target_class] in ids:
                                self.paths_to_target_data.append(row[0]) 
                            else:
                                self.paths_to_data.append(row[0])
                            self.labels[row[0]] = row[1]
                    else:
                        if name_to_id[target_class] in ids:
                            self.paths_to_target_data.append(row[0]) 
                        else:
                            self.paths_to_data.append(row[0])
                        self.labels[row[0]] = row[1]

        #Classification
        self.index_dict = make_index_dict(os.path.join(dir, "FSD50K.ground_truth", "class_labels_indices.csv"))

        # List of target classe(s)
        target_id = name_to_id[self.target_class]
        self.list_of_target_classes = [target_id]
        self.list_of_target_classes.extend(self.get_child_classes(self.ontology_dict[target_id]['child_ids']))
        self.list_of_target_classes = [self.ontology_dict[n]['name'] for n in self.list_of_target_classes]

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

    def get_child_classes(self, list_of_children):
        all_children = []
        for child in list_of_children:
            all_children.append(child)
            if children_of_child := self.get_child_classes(self.ontology_dict[child]['child_ids']):
                all_children.extend(children_of_child)

        return all_children

    def __len__(self):
        return len(self.paths_to_target_data)
    
    def __getitem__(self, idx):
        # The 16k samples were generated using sox
        # os.system('sox ' + basepath + audiofile+' -r 16000 ' + targetpath + audiofile + '> /dev/null 2>&1')
        wav_path = os.path.join(self.dir, "FSD50K.dev_audio" ,self.paths_to_target_data[idx] + ".wav") 

        x, file_sample_rate = torchaudio.load(wav_path)
        x = torchaudio.functional.resample(x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
        # TODO: number of seconds should be a parameter
        x = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=True)

        # Label for clasification
        if not self.supervised:
            classification_label = torch.zeros(200, device=self.device, dtype=torch.float32)
            for label_str in self.labels[self.paths_to_target_data[idx]].split(','):
                classification_label[int(self.index_dict[label_str])] = 1.0

        if self.rir:
            room = random.randint(0, len(self.rooms)-1)
            array_idx = random.randint(0, 5) * 5 
            source_list = []
            rir = self.select_rir(source_list, room, array_idx)
            x = self.apply_rir(rir, x[0])[None,0,:]

        mix = self.rms_normalize(x, True)

        isolated_sources = mix.clone()[None, ...]

        additionnal_idxs = []
        idxs_classes = [self.labels[self.paths_to_target_data[idx]]]
        for source_nb in range(self.max_sources-1):
            # TODO: set the probability to non-zero
            # Make sure that there is not nothing in the second mix
            if random.random() >= 0.5 or \
               (not self.supervised and source_nb == int(self.max_sources //2)):
                while True:
                    additionnal_idx = random.randint(0, len(self.paths_to_data)-1)
                    additionnal_idx_class = self.labels[self.paths_to_data[additionnal_idx]]
                    list_of_additionnal_classes = additionnal_idx_class.split(",")
                    add_idx = True
                    for cla in self.non_mixing_classes:
                        if cla in list_of_additionnal_classes:
                            add_idx = False
                    if add_idx and additionnal_idx_class not in idxs_classes and not additionnal_idx in additionnal_idxs:
                        additionnal_idxs.append(additionnal_idx)
                        idxs_classes.append(additionnal_idx_class)
                        break
            else:
                # Empty source
                additionnal_idxs.append(-1)
                idxs_classes.append("Nothing") 

        for index in additionnal_idxs:
            if index == -1:
                additionnal_x = torch.zeros_like(mix)
            else:
                wav_path = os.path.join(self.dir, "FSD50K.dev_audio" ,self.paths_to_data[index] + ".wav") 
                additionnal_x, file_sample_rate = torchaudio.load(wav_path)
                additionnal_x = torchaudio.functional.resample(additionnal_x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
                # TODO: number of seconds should be a parameter
                additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, self.nb_of_seconds, shuffle=True)

                if self.rir:
                    rir = self.select_rir(source_list, room, array_idx)
                    additionnal_x = self.apply_rir(rir, additionnal_x[0])[None,0,:]
                              
                additionnal_x = self.rms_normalize(additionnal_x, True)

                if torch.sum(torch.isnan(additionnal_x)):
                    print(self.paths_to_data[index] + ".wav")
                    additionnal_x = torch.zeros_like(additionnal_x)

                mix += additionnal_x

            isolated_sources = torch.cat((isolated_sources, additionnal_x[None, ...]))

        mix, factor = self.peak_normalize(mix)
        isolated_sources *= factor

        #randomize volume
        volume = random.random()

        mix *= volume
        isolated_sources *= volume

        if self.return_spectrogram:
            mix = self.stft(mix)
            isolated_sources = self.stft(isolated_sources)

        if not self.supervised:
            return mix, isolated_sources, idxs_classes, classification_label
        else:
            return mix, isolated_sources, idxs_classes
    
    def get_serialized_sample(self, idx, key=1500):
        wav_path = os.path.join(self.dir, "FSD50K.dev_audio" ,self.paths_to_target_data[idx] + ".wav") 

        x, file_sample_rate = torchaudio.load(wav_path)
        x = torchaudio.functional.resample(x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
        # TODO: number of seconds should be a parameter
        x = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=False)

        if self.rir:
            room = 0
            array_idx = 0 

            rir = self.rir_dataset['rir'][self.rooms[room]][self.sources[0]][()][:,array_idx:array_idx+5]
            rir = rir.transpose()
            rir = torch.tensor(rir).to(torch.float32)
            rir = torchaudio.functional.resample(rir, orig_freq=48000, new_freq=self.sample_rate).to(self.device)
            x = self.apply_rir(rir, x[0])[None,0,:]

        mix = self.rms_normalize(x, False)

        isolated_sources = mix.clone()[None, ...]

        additionnal_idxs = []
        idxs_classes = [self.labels[self.paths_to_target_data[idx]]]
        additionnal_idx = (idx*key) % len(self.paths_to_data)
        for source_nb in range(self.max_sources-1):
            if random.random() >= 0.5 or \
               (not self.supervised and source_nb == int(self.max_sources //2)):
                while True:
                    additionnal_idx += 1
                    additionnal_idx_class = self.labels[self.paths_to_data[additionnal_idx]]
                    list_of_additionnal_classes = additionnal_idx_class.split(",")
                    add_idx = True
                    for cla in self.non_mixing_classes:
                        if cla in list_of_additionnal_classes:
                            add_idx = False
                    if add_idx and additionnal_idx_class not in idxs_classes and not additionnal_idx in additionnal_idxs:
                        additionnal_idxs.append(additionnal_idx)
                        idxs_classes.append(additionnal_idx_class)
                        break
            else:
                # Empty source
                additionnal_idxs.append(-1)
                idxs_classes.append("Nothing") 

        additional_mix = torch.zeros_like(mix)
        for num_of_additionnal, index in enumerate(additionnal_idxs):
            if index == -1:
                additionnal_x = torch.zeros_like(additional_mix)
            else:
                wav_path = os.path.join(self.dir, "FSD50K.dev_audio" ,self.paths_to_data[index] + ".wav") 
                additionnal_x, file_sample_rate = torchaudio.load(wav_path)
                additionnal_x = torchaudio.functional.resample(additionnal_x, orig_freq=file_sample_rate, new_freq=self.sample_rate).to(self.device)
                # TODO: number of seconds should be a parameter
                additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, self.nb_of_seconds, shuffle=False)

                if self.rir:
                    rir = self.rir_dataset['rir'][self.rooms[room]][self.sources[num_of_additionnal+1]][()][:,array_idx:array_idx+5]
                    rir = rir.transpose()
                    rir = torch.tensor(rir).to(torch.float32)
                    rir = torchaudio.functional.resample(rir, orig_freq=48000, new_freq=self.sample_rate).to(self.device)
                    additionnal_x = self.apply_rir(rir, additionnal_x[0])[None,0,:]

                additionnal_x = self.rms_normalize(additionnal_x, False)
                
                mix += additionnal_x

            isolated_sources = torch.cat((isolated_sources, additionnal_x[None, ...]))

        mix, factor = self.peak_normalize(mix)
        isolated_sources *= factor

        volume = 0.75

        mix *= volume
        isolated_sources *= volume

        if self.return_spectrogram:
            mix = self.stft(mix)
            isolated_sources = self.stft(isolated_sources)

        return mix, isolated_sources, idxs_classes
    
    @staticmethod
    def get_right_number_of_samples(x, sample_rate, seconds, shuffle=False):
        nb_of_samples = seconds*sample_rate
        if x.shape[1] < nb_of_samples:
            missing_nb_of_samples = nb_of_samples-x.shape[1]
            if shuffle:
                random_number = random.randint(0, missing_nb_of_samples)
                x = torch.nn.functional.pad(x, (random_number, missing_nb_of_samples-random_number), mode="constant", value=0)
            else:
                if not missing_nb_of_samples % 2:
                    x = torch.nn.functional.pad(x, (int(missing_nb_of_samples/2), int(missing_nb_of_samples/2)), mode="constant", value=0)
                else:
                    x = torch.nn.functional.pad(x, (int(missing_nb_of_samples/2), int(missing_nb_of_samples/2)+1), mode="constant", value=0)
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
    def rms_normalize(x, augmentation=False):
        # Equation: 10*torch.log10((torch.abs(X)**2).mean()) = 0

        if augmentation:
            # Gain between -5 and 5 dB
            aug = torch.rand(1).item()*10 - 5
            augmentation_gain = 10 ** (aug/20)
        else:
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/(torch.abs(x)**2).mean()) 
       
        return augmentation_gain * normalize_gain * x
    
    @staticmethod
    def peak_normalize(x):
        factor = 1/torch.max(torch.abs(x))
        new_x = factor * x

        return new_x, factor


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    frame_size = 512
    hop_size = int(frame_size / 2)
    target_class = "Bark"
    dataset = FSD50KDataset("/home/jacob/dev/weakseparation/library/dataset/FSD50K",
                            frame_size, 
                            hop_size, 
                            target_class, 
                            forceCPU=True, 
                            return_spectrogram=False,
                            supervised=False,
                            isolated=True)
    print(len(dataset))

    # _, _, label = dataset.get_serialized_sample(10, 1300)

    # print(label)

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    for _ in range(100):
        for mix, isolatedSources, labels in dataloader:
            pass