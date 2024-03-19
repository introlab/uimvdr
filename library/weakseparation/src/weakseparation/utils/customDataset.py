import os
import torchaudio
import torch
import random

from .Windows import sqrt_hann_window
# from Windows import sqrt_hann_window
from torch.utils.data import Dataset

class DataEntry:
    def __init__(self, path, class_name, position) -> None:
        self.class_name = class_name
        self.path = path
        self.position = position

class CustomDataset(Dataset):
    """
        MCFSTD Dataset

        Args:
            dir (str): Directory for test dataset
            frame_size (int): Frame size for fft/ifft
            hop_size (int): Hop size for fft/ifft
            target_class (list of str): Classes that are considered for the target
            mic_array (str): respeaker, kinect or 16sounds
            sample_rate (int): sampling rate for audio samples
            max_sources (int): Maximum number of sources for mixing, should always be 2 or more
            forceCPU (bool): load with cpu or gpu
            supervised (bool): Whether to load the data for supervised or unsupervised training           
            return_spectorgram (bool): Whether the dataloaders should return spectrogram or waveforms
            nb_iteration (int): Number of iteration on the targets that should be done for 1 epoch
            nb_of_seconds (int): Number of seconds the audio sample should be
    """
    def __init__(self, 
                 dir,
                 frame_size, 
                 hop_size, 
                 target_class="Bark",
                 mic_array = "respeaker",
                 sample_rate=16000, 
                 max_sources=4, 
                 forceCPU=False,
                 supervised=True, 
                 return_spectrogram=True,
                 nb_iteration=1, 
                 nb_of_seconds=5) -> None:
        super().__init__()
        self.dir = dir
        self.sample_rate = sample_rate
        self.max_sources = max_sources
        self.target_class = target_class
        speech_set = {"Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking"}
        if not set(self.target_class).isdisjoint(speech_set):
            self.target_class = "Speech"
        self.nb_of_seconds = nb_of_seconds
        self.mic_array = mic_array
        self.supervised = supervised
        self.nb_iteration = nb_iteration
        if forceCPU:
            self.device = 'cpu'
        else:
            if (torch.cuda.is_available()):
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        self.return_spectrogram = return_spectrogram
        
        self.paths_to_data = []
        self.paths_to_target_data = []
        for root, dirs, files in os.walk(self.dir):
            for class_dir in dirs:
                class_dir_path = os.path.join(root, class_dir)
                current_mic_array = class_dir_path.split("/")[-3]
                if current_mic_array == self.mic_array:
                    for audio_file in os.scandir(class_dir_path):
                        if audio_file.is_file():
                            position = class_dir_path.split("/")[-2]
                            if not position == "Diffus1" and not position == "Diffus2":
                                if class_dir in self.target_class:
                                    if self.mic_array == "kinect" and (position == "A" or position == "B" or position == "C" or position == "D" or position == "E"):
                                        self.paths_to_target_data.append(DataEntry(audio_file,class_dir,position))
                                    elif self.mic_array != "kinect":
                                        self.paths_to_target_data.append(DataEntry(audio_file,class_dir,position))
                                else:
                                    self.paths_to_data.append(DataEntry(audio_file,class_dir,position))

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=frame_size,
            hop_length=hop_size,
            power=None,
            window_fn=sqrt_hann_window
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
        target_data: DataEntry = self.paths_to_target_data[idx] 

        x, _ = torchaudio.load(target_data.path)
        # Should I shuffle here?
        mix = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=False)

        additionnal_idxs = []
        idxs_classes = [target_data.class_name]
        for source_nb in range(self.max_sources-1):
            # Make sure that there is not nothing in the second mix
            if random.random() >= 0.5 or \
               (source_nb == int(self.max_sources //2)):
                while True:
                    additionnal_idx = random.randint(0, len(self.paths_to_data)-1)
                    additionnal_idx_class = self.paths_to_data[additionnal_idx].class_name
                    if additionnal_idx_class not in idxs_classes and \
                        not additionnal_idx in additionnal_idxs and \
                        self.paths_to_data[additionnal_idx].position != target_data.position :
                        additionnal_idxs.append(additionnal_idx)
                        idxs_classes.append(additionnal_idx_class)
                        break
            else:
                # Empty source
                additionnal_idxs.append(-1)
                idxs_classes.append("Nothing") 

        mix = self.rms_normalize(mix, True)

        isolated_sources = mix.clone()[None, ...]
        for index in additionnal_idxs:
            if index == -1:
                additionnal_x = torch.zeros_like(mix)
            else:
                wav_path = self.paths_to_data[index].path
                additionnal_x, file_sample_rate = torchaudio.load(wav_path)
                additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, self.nb_of_seconds, shuffle=False)

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
                idx (int): From 0 to length of dataset obtain with len(self)
                key (int): Number for getting the additional sources without randomness
        """
        idx = idx % len(self.paths_to_target_data)
        target_data: DataEntry = self.paths_to_target_data[idx] 

        x, _ = torchaudio.load(target_data.path)
        # Should I shuffle here?
        mix = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=False)

        additionnal_idxs = []
        idxs_classes = [target_data.class_name]
        additionnal_idx = (idx*key) % len(self.paths_to_data)
        for source_nb in range(self.max_sources-1):
            # Make sure that there is not nothing in the second mix
            if random.random() >= 0.5 or \
               (source_nb == int(self.max_sources //2)):
                while True:
                    additionnal_idx_class = self.paths_to_data[additionnal_idx].class_name
                    if additionnal_idx_class not in idxs_classes and \
                        not additionnal_idx in additionnal_idxs and \
                        self.paths_to_data[additionnal_idx].position != target_data.position :
                        additionnal_idxs.append(additionnal_idx)
                        idxs_classes.append(additionnal_idx_class)
                        break
                    additionnal_idx += 1
            else:
                # Empty source
                additionnal_idxs.append(-1)
                idxs_classes.append("Nothing") 
        
                additional_mix = torch.zeros_like(mix)

        isolated_sources = mix.clone()[None, ...]
        for index in additionnal_idxs:
            if index == -1:
                additionnal_x = torch.zeros_like(additional_mix)
            else:
                wav_path = self.paths_to_data[index].path
                additionnal_x, file_sample_rate = torchaudio.load(wav_path)
                additionnal_x = self.get_right_number_of_samples(additionnal_x, self.sample_rate, self.nb_of_seconds, shuffle=False)

                if torch.sum(torch.isnan(additionnal_x)):
                    print(self.paths_to_data[index].path)
                    additionnal_x = torch.zeros_like(additionnal_x)
                
                mix += additionnal_x

            isolated_sources = torch.cat((isolated_sources, additionnal_x[None, ...]))

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
                x = self.get_right_number_of_samples(x, self.sample_rate, self.nb_of_seconds, shuffle=False)
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
    def get_right_number_of_samples(x, sample_rate, seconds, shuffle=False):
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
    def rms_normalize(x, augmentation=False, gain=None):
        """
            Solves this equation: 10*torch.log10((torch.abs(x)**2).mean()) = 0

            Args:
                x (Tensor): Data to normalize
                augmentation (bool): Whether to normalize to 0 dB or a gain between -5 and 5 dB.
        """

        if augmentation:
            if gain is None:
                # Gain between -5 and 5 dB
                aug = torch.rand(1).item()*10 - 5
                augmentation_gain = 10 ** (aug/20)
            else:
                augmentation_gain = 10 ** (gain/20)
        else:
            # 0 dB
            augmentation_gain = 1
        
        normalize_gain  = torch.sqrt(1/((torch.abs(x)**2).mean()+torch.finfo(torch.float).eps)) 
    
        return augmentation_gain * normalize_gain * x
        
    @staticmethod
    def peak_normalize(x):
        """
            Peak normalization, the maximum now becomes 1 or -1

            Args:
                x (Tensor): Data to normalize
        """
        factor = 1/(torch.max(torch.abs(x))+torch.finfo(torch.float).eps)
        new_x = factor * x

        return new_x, factor



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    frame_size = 512
    hop_size = int(frame_size / 2)
    target_class = "Bark"
    mic_array = "16sounds"
    dataset = CustomDataset("/home/jacob/dev/weakseparation/library/dataset/Custom/separated",
                            frame_size, 
                            hop_size, 
                            target_class,
                            mic_array=mic_array, 
                            forceCPU=True, 
                            return_spectrogram=False)
    print(len(dataset))

    paths = [
        "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/A/Bark/4.wav",
        "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/D/Speech/10.wav",
        "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/G/Church_bell/15.wav",
        None
    ]
    dataset.get_personalized_sample(paths)

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    for _ in range(5):
        for mix, isolatedSources, labels in dataloader:
            pass