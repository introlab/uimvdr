import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir, batch_size, frame_size, hop_size, sample_rate=16000, max_sources=3, num_of_workers=4, return_spectrogram=True):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.max_sources = max_sources
        self.return_spectrogram = return_spectrogram

    def setup(self, stage: str):
        # Assign test dataset for use in dataloader(s)
        if stage == "fit":
            # TRAIN_VALIDATION_SPLIT = 0.9
            self.dataset_train = self.dataset(
                self.data_dir,
                self.frame_size,
                self.hop_size,
                type="train",
                sample_rate=self.sample_rate,
                max_sources=self.max_sources, 
                forceCPU=True,
                return_spectrogram = self.return_spectrogram,
            )
            self.dataset_val = self.dataset(
                self.data_dir, 
                self.frame_size, 
                self.hop_size, 
                type="val", 
                sample_rate=self.sample_rate,
                max_sources=self.max_sources, 
                forceCPU=True,
                return_spectrogram = self.return_spectrogram,
            )
            # self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset_val,
            #                                                         [int(len(
            #                                                             dataset_val) * TRAIN_VALIDATION_SPLIT),
            #                                                         int(len(dataset_val) - int(len(
            #                                                             dataset_val) * TRAIN_VALIDATION_SPLIT))])


        if stage == "validate":
            self.dataset_val = self.dataset(
                self.data_dir,
                self.frame_size,
                self.hop_size,
                type="val",
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                forceCPU=True,
                return_spectrogram = self.return_spectrogram,
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

