import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .Dataset import SeclumonsDataset

class SeclumonsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, frame_size, hop_size, sample_rate=16000, num_of_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

    def setup(self, stage: str):

        # Assign test dataset for use in dataloader(s)
        if stage == "fit":
            self.seclumons_train = SeclumonsDataset(self.data_dir, self.frame_size, self.hop_size, type="train", sample_rate=self.sample_rate, forceCPU=True)
            self.seclumons_val = SeclumonsDataset(self.data_dir, self.frame_size, self.hop_size, type="val", sample_rate=self.sample_rate, forceCPU=True)

        if stage == "validate":
            self.seclumons_val = SeclumonsDataset(self.data_dir, self.frame_size, self.hop_size, type="val", sample_rate=self.sample_rate, forceCPU=True)

        if stage == "predict":
            self.seclumons_val = SeclumonsDataset(self.data_dir, self.frame_size, self.hop_size, type="predict", sample_rate=self.sample_rate, forceCPU=True)

    def train_dataloader(self):
        return DataLoader(
            self.seclumons_train,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.seclumons_val,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.seclumons_val,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        )

