import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch

from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    """
        Class that dictates which dataset is used for every step (training, validation, test)

        Args:
            dataset (class): Class for train/val dataset
            data_dir (str): Directory for train/val dataset
            test_dataset (class): Class for test dataset
            test_data_dir (str): Directory for test dataset
            batch_size (int): Batch size for loading data
            frame_size (int): Frame size for fft/ifft
            hop_size (int): Hop size for fft/ifft
            target_class (list of str): Classes that are considered for the target
            non_mixing_classes (list of str): Classes that should not be included in the noise mixed with the targets
            branch_class (str): Used for detecting if the sample is isolated
            sample_rate (int): sampling rate for audio samples
            max_sources (int): Maximum number of sources for mixing, should always be 2 or more
            num_of_workers (int): Number of workers to load the data
            nb_of_seconds (int): Number of seconds the audio sample should be
            return_spectorgram (bool): Whether the dataloaders should return spectrogram or waveforms
            supervised (bool): Whether to load the data for supervised or unsupervised training
            isolated (bool): Whether the target should be isolated or not
    """
    def __init__(self, 
                 dataset,
                 data_dir,
                 test_dataset,
                 test_data_dir,
                 batch_size, 
                 frame_size, 
                 hop_size, 
                 target_class=None,
                 non_mixing_classes=None,
                 branch_class=None,
                 sample_rate=16000, 
                 max_sources=3, 
                 num_of_workers=4,
                 nb_of_seconds=5,
                 return_spectrogram=True,
                 supervised=True,
                 isolated=False):
        super().__init__()
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.test_data_dir = test_data_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.max_sources = max_sources
        self.return_spectrogram = return_spectrogram
        self.target_class = target_class
        self.non_mixing_classes = non_mixing_classes
        self.branch_class = branch_class
        self.supervised = supervised
        self.isolated = isolated
        self.nb_of_seconds=nb_of_seconds

    def setup(self, stage: str):
        """
            This is called before each stage to setup datasets. See pytorch lightning for more info.

            Args:
                stage (str): fit, val or test
        """
        nb_of_test_iteration = 10
        if stage == "fit":
            self.dataset_train = self.dataset(
                self.data_dir,
                self.frame_size,
                self.hop_size,
                type="train",
                target_class=self.target_class,
                non_mixing_classes=self.non_mixing_classes,
                branch_class=self.branch_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds, 
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
                isolated=self.isolated,
            )
            self.dataset_val = self.dataset(
                self.data_dir, 
                self.frame_size, 
                self.hop_size, 
                type="val",
                target_class=self.target_class,
                non_mixing_classes=self.non_mixing_classes,
                branch_class=self.branch_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                nb_iteration = 3,
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
                isolated=self.isolated,
            )
        if stage == "val":
            self.dataset_val = self.dataset(
                self.data_dir, 
                self.frame_size, 
                self.hop_size, 
                type="val",
                target_class=self.target_class,
                non_mixing_classes=self.non_mixing_classes,
                branch_class=self.branch_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                nb_iteration = 3,
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
                isolated=self.isolated,
            )
        if stage == "test":
            self.dataset_test_respeaker = self.test_dataset(
                self.test_data_dir, 
                self.frame_size, 
                self.hop_size, 
                target_class=self.target_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                supervised=self.supervised,
                mic_array="respeaker",  
                forceCPU=True, 
                return_spectrogram=False,
                nb_iteration = nb_of_test_iteration,
            )
            self.dataset_test_kinect = self.test_dataset(
                self.test_data_dir, 
                self.frame_size, 
                self.hop_size, 
                target_class=self.target_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,  
                supervised=self.supervised,
                mic_array="kinect",   
                forceCPU=True, 
                return_spectrogram=False,
                nb_iteration = nb_of_test_iteration,
            )
            self.dataset_test_16sounds = self.test_dataset(
                self.test_data_dir, 
                self.frame_size, 
                self.hop_size, 
                target_class=self.target_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                supervised=self.supervised,
                mic_array="16sounds",     
                forceCPU=True, 
                return_spectrogram=False,
                nb_iteration = nb_of_test_iteration,
            )
            self.dataset_test_fsd50k = self.dataset(
                self.data_dir, 
                self.frame_size, 
                self.hop_size, 
                type="test",
                target_class=self.target_class,
                non_mixing_classes=self.non_mixing_classes,
                branch_class=self.branch_class,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds, 
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
                isolated=True,
                nb_iteration = nb_of_test_iteration,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        )
    
    def test_dataloader(self):
        return [
        DataLoader(
            self.dataset_test_respeaker,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        ),
        DataLoader(
            self.dataset_test_kinect,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        ),
        DataLoader(
            self.dataset_test_16sounds,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        ),
        DataLoader(
            self.dataset_test_fsd50k,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        ),
        ]


