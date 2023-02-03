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


class UNet(pl.LightningModule):
    def __init__(self, in_channels, sources, mics):
        super().__init__()
        self.mics = mics
        self.sources = sources
        self.istft = InverseSpectrogram(
                n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
            )
        
        self.log_columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.epsilon = torch.finfo(torch.float).eps

        dim1 = 32
        dim2 = 64
        dim3 = 128
        dim4 = 256
        self.n_classes = sources

        self.down1 = nn.Sequential(
            nn.Conv2d(2*mics-1, dim1, (3, 3), padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(),
            nn.Conv2d(dim1, dim1, (3, 3), padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(dim1, dim2, (3, 3), padding=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),
            nn.Conv2d(dim2, dim2, (3, 3), padding=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(dim2, dim3, (3, 3), padding=1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(),
            nn.Conv2d(dim3, dim3, (3, 3), padding=1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(dim3, dim4, (3, 3), padding=1),
            nn.BatchNorm2d(dim4),
            nn.ReLU(),
            nn.Conv2d(dim4, dim3, (3, 3), padding=1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(),
            nn.ConvTranspose2d(dim3, dim3, (2, 2), stride=(2, 2), output_padding=(0,1)),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(dim3 + dim3, dim3, (3, 3), padding=1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(),
            nn.Conv2d(dim3, dim2, (3, 3), padding=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),
            nn.ConvTranspose2d(dim2, dim2, (2, 2), stride=(2, 2)),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(dim2+dim2, dim2, (3, 3), padding=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(),
            nn.Conv2d(dim2, dim1, (3, 3), padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim1, dim1, (2, 2), stride=(2, 2), output_padding=(1,0)),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(dim1+dim1, dim1, (3, 3), padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(),
            nn.Conv2d(dim1, dim1, (3, 3), padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(),
            nn.Conv2d(dim1, self.n_classes, (3, 3), padding=1),
        )

        self.save_hyperparameters()

    def forward(self, x):
        x = torch.real(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        bottleneck = self.bottleneck(down3)
        concat1 = torch.cat((bottleneck, down3), dim=1)
        up3 = self.up3(concat1)
        concat2 = torch.cat((up3, down2), dim=1)
        up2 = self.up2(concat2)
        concat3 = torch.cat((up2, down1), dim=1)
        output = self.up1(concat3)
        output = torch.sigmoid(output)

        return output

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch:
                mix 
                isolated sources (batch_num, sources, mics, freq, frame)
        """
        mix, isolatedSources, labels = batch

        isolatedSources = isolatedSources[:,:,0]

        input = torch.abs(mix).median(dim=1, keepdim=True).values
        phases = torch.angle(mix)

        for mic in range(1, self.mics):
            input = torch.cat((input, torch.cos(phases[:, 0] - phases[:, mic])[:, None, ...]), dim=1)
            input = torch.cat((input, torch.sin(phases[:, 0] - phases[:, mic])[:, None, ...]), dim=1)

        masks= self(input)

        pred = mix[:, 0][:, None, ...] * masks

        pit_loss, best_permutation = permutation_invariant_training(
            self.istft(pred),
            self.istft(isolatedSources),
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

        isolatedSources = isolatedSources[:,:,0]

        input = torch.real(mix).median(dim=1, keepdim=True).values
        phases = torch.angle(mix)

        for mic in range(1, self.mics):
            input = torch.cat((input, torch.cos(phases[:, 0] - phases[:, mic])[:, None, ...]), dim=1)
            input = torch.cat((input, torch.sin(phases[:, 0] - phases[:, mic])[:, None, ...]), dim=1)

        masks= self(input)

        pred = mix[:, 0][:, None, ...] * masks

        pit_loss, best_permutation = permutation_invariant_training(
            self.istft(pred),
            self.istft(isolatedSources),
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
            preds = pred[0]
            groundTruths = isolatedSources[0]
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