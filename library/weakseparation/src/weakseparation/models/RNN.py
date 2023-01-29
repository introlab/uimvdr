import torch
from torch import optim, nn
from torchmetrics.functional import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchaudio.transforms import InverseSpectrogram
import pytorch_lightning as pl
import wandb

from ..utils.Windows import cuda_sqrt_hann_window

import warnings
warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed as ")


class GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, mics):
        super().__init__()
        self.mics = mics
        self.BN = nn.BatchNorm2d(num_features=mics)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0)
        self.linear = nn.Linear(hidden_size, input_size) # self.fc = nn.Conv2d(in_channels=hidden_size, out_channels=input_size, kernel_size=1) 
        self.conv_sources = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,3), padding=1)
        self.sig = nn.Sigmoid()
        self.istft = InverseSpectrogram(
                n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
            )

        self.save_hyperparameters()

    def forward(self, x, hidden=None):

        original_shape = x.shape
        # N x M x F x T > N x M x F x T
        x = torch.real(x)

        # N x M x F x T > N x M x T x F
        x = x.permute(0, 1, 3, 2)

        # N x M x T x F > N x M x T x F
        x = self.BN(x)

        # N x M x F x T > N x T x F x M
        x = x.permute(0, 2, 3, 1)

        # N x T x F x M > N x T x F*M
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        # N x T x F*M > N x T x H
        x, h = self.GRU(x, hidden)

        # N x T x H > N x H x T
        x = x.permute(0, 2, 1)

        # N x H x T > N x H x T x 1
        x = torch.unsqueeze(x, 3)

        # N x H x T x 1 > N x 1 x T x H
        x = x.permute(0, 3, 2, 1)

        # N x 1 x T x H > N x S x T x H
        x = self.conv_sources(x)

        # N x S x T x H > N x S x T x F*M
        x = self.linear(x)

        #TODO: put nb of sources in variable
        x = torch.reshape(x, (original_shape[0], 3, original_shape[1], original_shape[2], original_shape[3]))
        
        # N x S x M x F x T > N x S x M x F x T 
        x = self.sig(x)
        return x, h

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch:
                mix 
                isolated sources (batch_num, sources, mics, freq, frame)
        """
        mix, isolatedSources, labels = batch

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        pit_loss, best_permutation = permutation_invariant_training(
            torch.abs(pred),
            torch.abs(isolatedSources),
            self.custom_mse,
            eval_func = "min"
        )

        pred = pit_permutate(pred, best_permutation)

        snr = scale_invariant_signal_noise_ratio(self.istft(pred), self.istft(isolatedSources)).mean()
        loss = pit_loss.sum()

        self.log("train_loss", loss)
        self.log('train_SI-SNR', snr)

        return loss

    def validation_step(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        # pit_loss, best_permutation = permutation_invariant_training(
        #     self.istft(pred)[:,:,0],
        #     self.istft(isolatedSources)[:,:,0],
        #     scale_invariant_signal_noise_ratio,
        #     eval_func = "max",
        # )
        # Need custom MSE to get the mse for each sample of the batch
        pit_loss, best_permutation = permutation_invariant_training(
            torch.abs(pred),
            torch.abs(isolatedSources),
            self.custom_mse,
            eval_func = "min",
        )

        pred = pit_permutate(pred, best_permutation)

        snr = scale_invariant_signal_noise_ratio(self.istft(pred), self.istft(isolatedSources)).mean()

        loss = pit_loss.sum()

        self.log('val_loss', loss)
        self.log('val_SI-SNR', snr)

        return loss

    @staticmethod
    def custom_mse(pred, target):
        loss = nn.functional.mse_loss(pred, target, reduction="none")
        loss = loss.mean((1,2,3))
        return loss

    def predict_step(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        return pred, isolatedSources, labels

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer