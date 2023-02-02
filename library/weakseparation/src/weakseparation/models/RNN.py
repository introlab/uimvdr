import torch
from torch import optim, nn
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
    def __init__(self, input_size, hidden_size, num_layers, mics):
        super().__init__()
        self.mics = mics
        self.BN = nn.BatchNorm2d(num_features=mics)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0)
        self.linear = nn.Linear(hidden_size, hidden_size) # self.fc = nn.Conv2d(in_channels=hidden_size, out_channels=input_size, kernel_size=1) 
        self.sig = nn.Sigmoid()
        self.istft = InverseSpectrogram(
                n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
            )
        
        self.log_columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.epsilon = torch.finfo(torch.float).eps

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

        # N x T x H  > N x T x H
        x = nn.functional.relu(x)

        # N x T x H  > N x 1 x T x H
        x = x[:,None,...]

        # N x 1 x T x H > N x 1 x T x H (H=F*2)
        x = self.linear(x)

        #TODO: put nb of sources in variable
        x = torch.reshape(x, (original_shape[0], 2, original_shape[1], original_shape[2], original_shape[3]))
        
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

        mix = mix[:,0][:, None, ...]
        isolatedSources = isolatedSources[:,:,0][:, :, None, ...]

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        pit_loss, best_permutation = permutation_invariant_training(
            self.istft(pred)[:,:,0],
            self.istft(isolatedSources)[:,:,0],
            scale_invariant_signal_distortion_ratio,
            eval_func = "max",
        )
        # pit_loss, best_permutation = permutation_invariant_training(
        #     pred,
        #     isolatedSources,
        #     self.custom_mse,
        #     eval_func = "min"
        # )

        pred = pit_permutate(pred, best_permutation)

        pred_waveform = self.istft(pred)
        pred_isolatedSources = self.istft(isolatedSources)

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, pred_isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, pred_isolatedSources).mean()
        # loss = pit_loss.sum()
        loss = pit_loss.mean()

        self.log("train_loss", loss)
        self.log('train_SNR', snr)
        self.log('train_SI-SDR', sdr)

        return loss

    def validation_step(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        mix = mix[:,0][:, None, ...]
        isolatedSources = isolatedSources[:,:,0][:, :, None, ...]

        masks, h = self(mix)

        pred = torch.einsum("iklm,ijklm->ijklm", mix, masks)

        pit_loss, best_permutation = permutation_invariant_training(
            self.istft(pred)[:,:,0],
            self.istft(isolatedSources)[:,:,0],
            scale_invariant_signal_distortion_ratio,
            eval_func = "max",
        )
        # Need custom MSE to get the mse for each sample of the batch
        # pit_loss, best_permutation = permutation_invariant_training(
        #     pred,
        #     isolatedSources,
        #     self.custom_mse,
        #     eval_func = "min",
        # )

        pred = pit_permutate(pred, best_permutation)

        pred_waveform = self.istft(pred)
        pred_isolatedSources = self.istft(isolatedSources)

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, pred_isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, pred_isolatedSources).mean()

        # loss = pit_loss.sum()
        loss = pit_loss.mean()

        if not self.current_epoch % 49 and batch_idx == 0 and self.logger is not None:
            preds = pred[0,:,0]
            groundTruths = isolatedSources[0,:,0]
            label = labels[0]
            i = 0
            table = []
            for source, groundTruth, id in zip(preds, groundTruths, label):
                waveform_source = self.istft(source)
                waveform_groundTruth = self.istft(groundTruth)
                class_label = id_to_class[id.item()]
                table.append([
                    class_label,
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title=class_label)),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title=class_label)),
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