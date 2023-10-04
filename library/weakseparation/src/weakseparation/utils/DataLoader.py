import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch

from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
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
                 sample_rate=16000, 
                 max_sources=3, 
                 num_of_workers=4, 
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
        self.supervised = supervised
        self.isolated = isolated

    def setup(self, stage: str):
        nb_of_test_iteration = 5
        if stage == "fit":
            self.dataset_train = self.dataset(
                self.data_dir,
                self.frame_size,
                self.hop_size,
                type="train",
                target_class=self.target_class,
                non_mixing_classes=self.non_mixing_classes,
                sample_rate=self.sample_rate,
                max_sources=self.max_sources, 
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
            sample_rate=self.sample_rate,
            max_sources=self.max_sources, 
            forceCPU=True,
            return_spectrogram=self.return_spectrogram,
            supervised=self.supervised,
            isolated=self.isolated,
        )

        self.dataset_test_respeaker = self.test_dataset(
            self.test_data_dir, 
            self.frame_size, 
            self.hop_size, 
            target_class=self.target_class,
            sample_rate=self.sample_rate,
            max_sources=self.max_sources,
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
            sample_rate=self.sample_rate,
            max_sources=self.max_sources, 
            forceCPU=True,
            return_spectrogram=self.return_spectrogram,
            supervised=self.supervised,
            isolated=self.isolated,
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


