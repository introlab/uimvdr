import os
import json
import csv
import torchaudio
import torch
import random

from .Windows import sqrt_hann_window
# from Windows import sqrt_hann_window
from torch.utils.data import Dataset
from .customDataset import DataEntry

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

def get_tree(index_dict, data):
    node_dict = {label['id']: OntNode(label, index_dict) for label in data}

    for node_id in node_dict.keys():
        node = node_dict[node_id]
        for child_id in node.child_ids:
            child_node = node_dict[child_id]
            node.add_child(child_node)
            child_node.add_parent(node)

    root = OntNode.root(index_dict)
    for node_id in node_dict.keys():
        node = node_dict[node_id]
        if len(node.parents()) == 0:
            root.add_child(node)
            node.add_parent(root)

    return (root, node_dict)

class OntNode():
    def __init__(self, label_dict, index_dict):
        self.id = label_dict['id']
        if self.id in index_dict:
            self.index = int(index_dict[self.id])
        else:
            self.index = None
        self.name = label_dict['name']
        self.description = label_dict['description']
        self.citation_uri = label_dict['citation_uri']
        self.child_ids = label_dict['child_ids']
        self.parent_ids = []
        self.child_nodes = []
        self.parent_nodes = []
        self.restrictions = label_dict['restrictions']

    @classmethod
    def root(cls, index_dict):
        label_dict = {'id': 'ROOT', 'name': 'ROOT', 'description':
                      'ROOT', 'citation_uri': '', 'child_ids': [],
                      'restrictions': ''}
        return cls(label_dict, index_dict)

    def add_parent(self, parent_node):
        if not self.is_root():
            self.parent_nodes.append(parent_node)
            self.parent_nodes.extend(parent_node.parent_nodes)
            self.parent_ids.append(parent_node.id)
            self.parent_ids.extend(parent_node.parent_ids)
        return None

    def add_child(self, child_node):
        self.child_nodes.append(child_node)
        return None

    def parents(self):
        if not self.is_root():
            return self.parent_nodes
        return None

    def children(self):
        return self.child_nodes

    def is_root(self):
        root = (self.name == "ROOT")
        if root:
            print("Node is root")
        return root


class DroneAudioSetDataset(Dataset):
    """
        AudioSet Dataset.

        Args:
            dir (str): Directory for test dataset
            frame_size (int): Frame size for fft/ifft
            hop_size (int): Hop size for fft/ifft
            target_class (list of str): Classes that are considered for the target
            non_mixing_classes (list of str): Classes that should not be in noise
            branch_class (str): Isolated target needs to come from this branch class, useful when target is less specific like speech
            type (str): train, val or test
            sample_rate (int): sampling rate for audio samples
            max_sources (int): Maximum number of sources for mixing, should always be 2 or more
            forceCPU (bool): load with cpu or gpu
            supervised (bool): Whether to load the data for supervised or unsupervised training           
            nb_iteration (int): Number of iteration on the targets that should be done for 1 epoch
            nb_of_seconds (int): Number of seconds the audio sample should be
    """
    def __init__(self,
                 drone_dataset_dir, 
                 audioset_dir,
                 frame_size, 
                 hop_size, 
                 type="train", 
                 sample_rate=16000, 
                 max_sources=3, 
                 forceCPU=False, 
                 return_spectrogram=True, 
                 supervised=True,
                 nb_iteration=1,
                 nb_of_seconds=3) -> None:
        super().__init__()
        self.drone_dataset_dir = drone_dataset_dir
        self.audioset_dir = audioset_dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.nb_of_seconds = nb_of_seconds
        self.type = type
        self.return_spectrogram = return_spectrogram
        self.supervised = supervised
        self.nb_iteration = nb_iteration
        self.ontology_dict = {}
        ontology = json.load(open(os.path.join(audioset_dir, "ontology", "ontology.json")))
        for item in ontology:
            self.ontology_dict[item["id"]] = item

        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        self.paths_to_data = []
        self.paths_to_target_data = []
        self.labels = {}
        if self.type == "train":
            csv_path = os.path.join(audioset_dir, "pointer_files", "whole_train_data.json")
        elif self.type == "val":
            csv_path = os.path.join(audioset_dir, "pointer_files", "eval_data.json")
        else:
            raise RuntimeError(f"Dataset does not support type {self.type}")
        
        drone_dir = os.path.join(drone_dataset_dir, self.type)
        for root, _, files in os.walk(drone_dir):
            for filename in files:
                wav_path = os.path.join(root, filename)
                self.paths_to_target_data.append(wav_path)
                self.labels[wav_path] = "Drone"

        with open(csv_path, mode='r') as json_file:
            data_dict = json.load(json_file)
            for data in data_dict["data"]:
                ids = set(data["labels"].split(","))
                wav_path = "/".join((data["wav"].split("/")[4:]))
                self.paths_to_data.append(wav_path)
                label_list = [self.ontology_dict[label]["name"] for label in data["labels"].split(",")]
                label_str = ",".join(label_list)
                self.labels[wav_path] = label_str
        
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
        """
            Number of items in dataset
        """
        return len(self.paths_to_target_data)*self.nb_iteration
    
    def __getitem__(self, idx):
        """
            Getter for data in data set.

            Args:
                idx (int): From 0 to lenght of dataset obtain with len(self)
        """
        idx = idx % len(self.paths_to_target_data)

        if self.type != "test":
            wav_path = os.path.join(self.drone_dataset_dir, self.paths_to_target_data[idx])
        else:
            wav_path = os.path.join(self.drone_dataset_dir, self.paths_to_target_data[idx])

        x, file_sample_rate = torchaudio.load(wav_path)

        # Random channel
        x = x[random.randint(0, x.shape[0]-1)][None, ...]

        x = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=True)

        mix = self.rms_normalize(x, True)

        isolated_sources = mix.clone()[None, ...]

        additionnal_idxs = []
        idxs_classes = [self.labels[self.paths_to_target_data[idx]]]
        for source_nb in range(self.max_sources-1):
            # Make sure that there is not nothing in the second mix
            if random.random() >= 0.5 or \
               ((not self.supervised or self.type=="test" or self.type=="val") and source_nb == int(self.max_sources // 2)):
                while True:
                    additionnal_idx = random.randint(0, len(self.paths_to_data)-1)
                    additionnal_idx_class = self.labels[self.paths_to_data[additionnal_idx]]
                    if additionnal_idx_class not in idxs_classes and not additionnal_idx in additionnal_idxs:
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
                if self.type != "test":
                    wav_path = os.path.join(self.audioset_dir, self.paths_to_data[index])
                else:
                    wav_path = os.path.join(self.audioset_dir, self.paths_to_data[index])

                additionnal_x, file_sample_rate = torchaudio.load(wav_path)
                additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, self.nb_of_seconds, shuffle=True)
                              
                additionnal_x = self.rms_normalize(additionnal_x, True)

                mix += additionnal_x

            isolated_sources = torch.cat((isolated_sources, additionnal_x[None, ...]))

        mix, factor = self.peak_normalize(mix)
        isolated_sources *= factor

        #randomize volume
        volume = random.uniform(0.1,1)

        mix *= volume
        isolated_sources *= volume

        if self.return_spectrogram:
            mix = self.stft(mix)
            isolated_sources = self.stft(isolated_sources)

        return mix, isolated_sources, idxs_classes
    
    def get_serialized_sample(self, idx, key=1500):
        """
            Getter that doesn't have radomness for logging

            Args:
                idx (int): From 0 to lenght of dataset obtain with len(self)
                key (int): Number for getting the additional sources without randomness
        """
        idx = idx % len(self.paths_to_target_data)
        wav_path = os.path.join(self.drone_dataset_dir, self.paths_to_target_data[idx]) 

        x, file_sample_rate = torchaudio.load(wav_path)

        # First channel
        x = x[0][None, ...]

        x = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=False)

        mix = self.rms_normalize(x, False)

        isolated_sources = mix.clone()[None, ...]

        additionnal_idxs = []
        idxs_classes = [self.labels[self.paths_to_target_data[idx]]]
        additionnal_idx = (idx*key) % len(self.paths_to_data)
        for source_nb in range(self.max_sources-1):
            if random.random() >= 0.5 or \
               ((not self.supervised or self.type=="test" or self.type=="val") and source_nb == int(self.max_sources //2)):
                while True:
                    additionnal_idx += 1
                    additionnal_idx_class = self.labels[self.paths_to_data[additionnal_idx]]
                    if additionnal_idx_class not in idxs_classes and not additionnal_idx in additionnal_idxs:
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
                wav_path = os.path.join(self.audioset_dir, self.paths_to_data[index]) 
                additionnal_x, file_sample_rate = torchaudio.load(wav_path)
                additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, self.nb_of_seconds, shuffle=False)

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
    
    def get_personalized_sample(self, paths: list[str]):
        """
            Generate an example with the paths

            Args:
                paths (list[str]): Paths to use for the example. Need to have atleast one path in the first index. Pass None in the list to have zeros. 
        """
        idxs_classes = []
        mix = None
        isolated_sources = None

        for path in paths:
            if path is not None:
                split_path = path.split("/")
                data = DataEntry(path, split_path[-2], split_path[-3])
                idxs_classes.append(data.class_name)
                x, _ = torchaudio.load(data.path)
                x = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=False, pad_end=True)
                x = self.rms_normalize(x, False)
            else:
                x = torch.zeros_like(mix)
                idxs_classes.append("Nothing")

            
            if mix is None:
                mix = x
            else:
                mix += x

            if isolated_sources is None:
                isolated_sources = mix.clone()[None, ...]
            else:
                isolated_sources = torch.cat((isolated_sources, x[None, ...]))
        
        mix, factor = self.peak_normalize(mix)
        isolated_sources *= factor

        if self.return_spectrogram:
            mix = self.stft(mix)
            isolated_sources = self.stft(isolated_sources)

        return mix, isolated_sources, idxs_classes
    
    @staticmethod
    def get_right_number_of_samples(x, sample_rate, seconds, shuffle=False, pad_end=False):
        """
            If the waveform is too short pad it to make it the nubmer of seconds desired else if it's too long
            cut it.

            Args:
                x (Tensor): Waveform with shape (..., time)
                sample_rate (int): Sample rate of the waveform
                seconds (int): Desired number of seconds
                shuffle (bool): If cutting, select radomn a random segment else take the first segment. If padding pad 
                                randomly each side else pad equally.
        """
        nb_of_samples = seconds*sample_rate
        if x.shape[1] < nb_of_samples:
            missing_nb_of_samples = nb_of_samples-x.shape[1]
            if shuffle:
                random_number = random.randint(0, missing_nb_of_samples)
                x = torch.nn.functional.pad(x, (random_number, missing_nb_of_samples-random_number), mode="constant", value=0)
            else:
                if pad_end:
                    x = torch.nn.functional.pad(x, (0, int(missing_nb_of_samples)), mode="constant", value=0)
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
    def rms_normalize(x, augmentation=False):
        """
            Solves this equation: 10*torch.log10((torch.abs(x)**2).mean()) = 0

            Args:
                x (Tensor): Data to normalize
                augmentation (bool): Whether to normalize to 0 dB or a gain between -10 and 10 dB.
        """

        if augmentation:
            # Gain between -10 and 10 dB
            aug = torch.rand(1).item()*20 - 10
            augmentation_gain = 10 ** (aug/20)
        else:
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/((torch.abs(x)**2).mean()+torch.finfo(torch.float).eps)) 
       
        return augmentation_gain * normalize_gain * x
    
    @staticmethod
    def peak_normalize(x):
        factor = 1/(torch.max(torch.abs(x))+torch.finfo(torch.float).eps)
        new_x = factor * x

        return new_x, factor


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    frame_size = 512
    hop_size = int(frame_size / 2)

    dataset = DroneAudioSetDataset(
                            "/home/jacob/dev/weakseparation/library/dataset/drone_dataset",
                            "/media/jacob/2fafdbfa-bd75-431c-abca-c664f105eef9/audioset",
                            frame_size, 
                            hop_size, 
                            forceCPU=True, 
                            return_spectrogram=False,
                            type="val",
                            supervised=False)
    print(len(dataset))

    # _, _, label = dataset.get_serialized_sample(10, 1300)

    # print(label)

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    for _ in range(100):
        for mix, isolatedSources, labels in dataloader:
            pass