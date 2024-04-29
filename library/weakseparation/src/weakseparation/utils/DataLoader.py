import pytorch_lightning as pl
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
                 batch_size, 
                 frame_size, 
                 hop_size,
                 target_class = "drone", 
                 sample_rate=16000, 
                 max_sources=3, 
                 num_of_workers=4,
                 nb_of_seconds=5,
                 return_spectrogram=True,
                 supervised=True):
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
        self.supervised = supervised
        self.nb_of_seconds=nb_of_seconds
        self.target_class = target_class

    def setup(self, stage: str):
        """
            This is called before each stage to setup datasets. See pytorch lightning for more info.

            Args:
                stage (str): fit, val or test
        """
        if stage == "fit":
            self.dataset_train = self.dataset(
                self.data_dir,
                self.frame_size,
                self.hop_size,
                target_class=self.target_class,
                type="train",
                sample_rate=self.sample_rate,
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds, 
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
            )
            self.dataset_val = self.dataset(
                self.data_dir, 
                self.frame_size, 
                self.hop_size,
                target_class=self.target_class, 
                sample_rate=self.sample_rate,
                type="val",
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                nb_iteration = 3,
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
            )
        if stage == "val":
            self.dataset_val = self.dataset(
                self.data_dir,
                self.frame_size, 
                self.hop_size,
                target_class=self.target_class,
                sample_rate=self.sample_rate, 
                type="val",
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                nb_iteration = 3,
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
            )
        if stage == "test":
            self.dataset_test = self.dataset(
                self.data_dir,
                self.frame_size, 
                self.hop_size,
                target_class=self.target_class,
                sample_rate=self.sample_rate, 
                type="test",
                max_sources=self.max_sources,
                nb_of_seconds=self.nb_of_seconds,
                nb_iteration = 100,
                forceCPU=True,
                return_spectrogram=self.return_spectrogram,
                supervised=self.supervised,
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
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=False,
            persistent_workers=False,
        )


