import torch
from torch import optim, nn
from asteroid.losses import MixITLossWrapper, multisrc_mse
from torchmetrics.functional import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio import signal_noise_ratio, scale_invariant_signal_distortion_ratio
from torchaudio.transforms import InverseSpectrogram
import pytorch_lightning as pl
import wandb

from ..utils.Windows import cuda_sqrt_hann_window
from ..utils.LabelUtils import id_to_class
from ..utils.PlotUtils import plot_spectrogram_from_waveform

import warnings
warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed as ")


class GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, mics, sources, supervised=True):
        super().__init__()
        self.mics = mics
        self.sources = sources
        self.supervised = supervised
        self.BN = nn.BatchNorm2d(num_features=mics)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0)
        self.conv = nn.Conv2d(in_channels=mics, out_channels=self.sources, kernel_size=3, padding=1)
        self.linear = nn.Linear(2*hidden_size, input_size//2) # self.fc = nn.Conv2d(in_channels=hidden_size, out_channels=input_size, kernel_size=1) 
        self.sig = nn.Sigmoid()
        self.istft = InverseSpectrogram(
                n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
            )
        
        self.log_columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.epsilon = torch.finfo(torch.float).eps
        self.mixit_loss = MixITLossWrapper(multisrc_mse, generalized=False)

        self.save_hyperparameters()

    def forward(self, x, hidden=None):

        original_shape = x.shape
        # N x M x F x T > N x M x F x T
        x = torch.abs(x)

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

        # N x T x H  > N x T x H
        x = nn.functional.relu(x)

        # N x T x H  > N x 1 x T x H
        x = x[:,None,...]

        x = self.conv(x)

        x = nn.functional.relu(x) 

        # N x 1 x T x H > N x 1 x T x H (H=F*2)
        x = self.linear(x)

        #TODO: put nb of sources in variable
        x = torch.reshape(x, (original_shape[0], self.sources, original_shape[2]//2, original_shape[3]))
        
        # N x S x M x F x T > N x S x M x F x T 
        x = self.sig(x)
        return x, h

    def training_step(self, batch, batch_idx):
        if self.supervised:
            loss = self.training_step_supervised(batch, batch_idx)
        else:
            loss = self.training_step_unsupervised(batch, batch_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.supervised:
            loss = self.validation_step_supervised(batch, batch_idx)
        else:
            loss = self.validation_step_unsupervised(batch, batch_idx)

        return loss

    def training_step_supervised(self, batch, batch_idx):
        """
        Args:
            batch:
                mix 
                isolated sources (batch_num, sources, mics, freq, frame)
        """
        mix, isolatedSources, labels  = batch
    
        # check the effect of median
        input_real = torch.real(mix).median(dim=1, keepdim=True).values
        input_img = torch.imag(mix).median(dim=1, keepdim=True).values
        input  = torch.concat((input_real, input_img), dim=2)

        masks, hidden = self(input)

        pred = mix[:, 0][:, None, ...] * masks

        target = torch.abs(isolatedSources[:, :, 0])
        loss, min_idx, parts = self.mixit_loss(
            torch.abs(pred),
            target,
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(pred, isolatedSources[:, :, 0], min_idx, parts)
        pred_waveform = self.istft(pred)
        target_waveform = self.istft(isolatedSources[:, :, 0])
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, target_waveform).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()


        self.log("train_loss", loss)
        self.log('train_SNR', snr)
        self.log('train_SI-SDR', sdr)

        return loss
    
    def validation_step_supervised(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        # mix isolated sources here to always have the same samples
        # target = isolatedSources.reshape((-1, self.sources, isolatedSources.shape[2], isolatedSources.shape[3], isolatedSources.shape[4]))
        # mix = torch.sum(target, dim=1)

        # check the effect of median
        input_real = torch.real(mix).median(dim=1, keepdim=True).values
        input_img = torch.imag(mix).median(dim=1, keepdim=True).values
        input  = torch.concat((input_real, input_img), dim=2)

        masks, hidden = self(input)

        pred = mix[:, 0][:, None, ...] * masks

        loss, min_idx, parts = self.mixit_loss(
            torch.abs(pred),
            torch.abs(isolatedSources[:, :, 0]),
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(pred, isolatedSources[:, :, 0], min_idx, parts)
        pred_waveform = self.istft(pred)
        target_waveform = self.istft(isolatedSources[:, :, 0])

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, target_waveform).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()

        if not self.current_epoch % 25 and batch_idx == 0 and self.logger is not None:
            preds = pred_waveform[0]
            groundTruths = target_waveform[0]
            #label = labels[:,0]
            i = 0
            table = []
            # for waveform_source, waveform_groundTruth, id in zip(preds, groundTruths, label):
            for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                #class_label = id_to_class[id.item()]#[id_to_class[item.item()] for item in id]
                table.append([
                    f"source {i}",
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                ])
                i+=1
            self.logger.log_table(key="results", columns=self.log_columns, data=table)


        self.log('val_loss', loss)
        self.log('val_SNR', snr)
        self.log('val_SI-SDR', sdr)

        return loss

    @staticmethod
    def custom_mse(pred, target):
        phase_sensitive_target = torch.abs(target) * torch.cos(torch.angle(target) - torch.angle(pred))
        loss = nn.functional.mse_loss(torch.abs(pred), phase_sensitive_target, reduction="none")
        loss = loss.mean((1,2,3))
        return loss

    def predict_step(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        return pred, isolatedSources, labels

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, maximize=True)
        return optimizer