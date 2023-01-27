import torch
from torch import optim, nn
from torchmetrics.functional import permutation_invariant_training
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchaudio.transforms import InverseSpectrogram
import pytorch_lightning as pl

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
            nn.functional.mse_loss,
            eval_func = "min"
        )

        # pred = self.reorder_source(isolatedSources, best_permutation)
        loss = pit_loss.sum()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        pit_loss, best_permutation = permutation_invariant_training(
            torch.abs(pred),
            torch.abs(isolatedSources),
            nn.functional.mse_loss,
            eval_func = "min"
        )

        # pred = self.reorder_source(isolatedSources, best_permutation)
        loss = pit_loss.sum()

        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        return pred 

    #Taken from Asteroid
    @staticmethod
    def reorder_source(source, batch_indices):
        """ Reorder sources according to the best permutation.
        Args:
            source (torch.Tensor): Tensor of shape [batch, n_src, time]
            batch_indices (torch.Tensor): Tensor of shape [batch, n_src].
                Contains optimal permutation indices for each batch.
        Returns:
            :class:`torch.Tensor`:
                Reordered sources of shape [batch, n_src, time].
        """
        reordered_sources = torch.stack(
            [torch.index_select(s, 0, b) for s, b in zip(source, batch_indices)]
        )
        return reordered_sources

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer