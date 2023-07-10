import torch
from asteroid.losses import MixITLossWrapper, multisrc_mse
from torch import optim, nn
from torchmetrics.functional.audio import signal_noise_ratio, scale_invariant_signal_distortion_ratio
from torchaudio.transforms import InverseSpectrogram, Spectrogram
import pytorch_lightning as pl
import wandb

from ..utils.Windows import cuda_sqrt_hann_window
from ..utils.LabelUtils import id_to_class
from ..utils.PlotUtils import plot_spectrogram_from_waveform
from .causal_sudormrf import CausalSuDORMRF
from .improved_sudormrf import ImprovedSuDORMRF

class SudoRmRf(pl.LightningModule):
    def __init__(self, in_channels, sources, mics, n_fft=512, supervised=True):
        super().__init__()
        self.mics = mics
        self.sampling_rate = 16000
        self.sources = sources
        self.istft = InverseSpectrogram(
                n_fft=n_fft, hop_length=int(n_fft/2), window_fn=cuda_sqrt_hann_window
            )

        self.mixit_loss = MixITLossWrapper(self.negative_si_sdr, generalized=False)
        # self.mixit_loss = MixITLossWrapper(multisrc_mse, generalized=False)

        self.log_columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.epsilon = torch.finfo(torch.float).eps
        self.supervised = supervised

        # self.model = CausalSuDORMRF(
        #     in_audio_channels=in_channels,
        #     out_channels=256,
        #     in_channels=512,
        #     num_blocks=4,
        #     upsampling_depth=5,
        #     enc_kernel_size=21,
        #     enc_num_basis=512,
        #     num_sources=sources
        # )
        self.model = ImprovedSuDORMRF(
            out_channels=256,
            in_channels=512,
            num_blocks=4,
            upsampling_depth=5,
            enc_kernel_size=21,
            enc_num_basis=512,
            num_sources=sources
        )

        self.save_hyperparameters()

    def forward(self, x):
        output = self.model(x)
        output = torch.tanh(output)

        return output
    
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

        # if isolatedSources.shape[0] % 2:
        #     isolatedSources = isolatedSources[:-1]

        # target_waveform = isolatedSources.reshape(-1, self.sources, isolatedSources.shape[-1])
        # input = torch.sum(target_waveform, dim=1, keepdim=True)

        pred = self(mix)

        isolatedSources = isolatedSources[:, :, 0]
        loss, min_idx, parts = self.mixit_loss(
            pred,
            isolatedSources,
            return_est=True
        )

        pred_waveform = self.mixit_loss.reorder_source(pred, isolatedSources, min_idx, parts)
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, isolatedSources).mean()


        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log('train_SNR', snr, on_step=False, on_epoch=True)
        self.log('train_SI-SDR', sdr, on_step=False, on_epoch=True)

        return loss

    def validation_step_supervised(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        # mix isolated sources here to always have the same samples
        # if isolatedSources.shape[0] % 2:
        #     isolatedSources = isolatedSources[:-1]

        # target_waveform = isolatedSources.reshape(-1, self.sources, isolatedSources.shape[-1])
        # input = torch.sum(target_waveform, dim=1, keepdim=True)

        pred = self(mix)

        isolatedSources = isolatedSources[:, :, 0]
        loss, min_idx, parts = self.mixit_loss(
            pred,
            isolatedSources,
            return_est=True
        )

        pred_waveform = self.mixit_loss.reorder_source(pred, isolatedSources, min_idx, parts)

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, isolatedSources).mean()

        if not self.current_epoch % 10 and batch_idx == 0 and self.logger is not None:
            preds = pred_waveform[0]
            groundTruths = isolatedSources[0]
            label = labels[:,0]
            i = 0
            table = []
            for waveform_source, waveform_groundTruth, id in zip(preds, groundTruths, label):
                class_label = id_to_class[id.item()]
                table.append([
                    class_label,
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, self.sampling_rate, title=class_label)),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, self.sampling_rate, title=class_label)),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=self.sampling_rate),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=self.sampling_rate),
                ])
                i+=1
            self.logger.log_table(key="results", columns=self.log_columns, data=table)


        self.log('val_loss', loss)
        self.log('val_SNR', snr)
        self.log('val_SI-SDR', sdr)

        return loss

    def training_step_unsupervised(self, batch, bactch_idx):
        """
        Args:
            batch:
                mix (batch_num, mics, time)
                isolated sources (batch_num, sources, mics, time)
        """
        mix, isolatedSources, labels  = batch

        isolatedSources = isolatedSources[:,:,0]
        mix = mix[:,0][:,None,...]

        target_waveform = isolatedSources.reshape(-1, int(self.sources/2), int(self.sources/2), isolatedSources.shape[-1])
        target_waveform = torch.sum(target_waveform, dim=2, keepdim=False)

        pred = self(mix)

        # Mixture consistency projection
        # consistent_pred = torch.zeros_like(pred)
        # for idx in range(self.sources):
        #     consistent_pred[:, idx] = pred[:, idx] + (mix[:,0] - (torch.sum(pred, dim=1)-pred[:, idx]))/self.sources
        # pred = consistent_pred

        loss, min_idx, parts = self.mixit_loss(
            pred,
            target_waveform,
            return_est=True
        )

        pred_waveform = self.mixit_loss.reorder_source(pred, target_waveform, min_idx, parts)
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, target_waveform).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log('train_SNR', snr, on_step=False, on_epoch=True)
        self.log('train_SI-SDR', sdr, on_step=False, on_epoch=True)

        return loss
    
    def validation_step_unsupervised(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        isolatedSources = isolatedSources[:,:,0]
        mix = mix[:,0][:,None,...]

        labels = labels.reshape(-1, int(self.sources/2), int(self.sources/2))

        target_waveform = isolatedSources.reshape(-1, int(self.sources/2), int(self.sources/2), isolatedSources.shape[-1])
        target_waveform = torch.sum(target_waveform, dim=2, keepdim=False)

        pred = self(mix)

        # Mixture consistency projection
        # consistent_pred = torch.zeros_like(pred)
        # for idx in range(self.sources):
        #     consistent_pred[:, idx] = pred[:, idx] + (mix[:,0] - (torch.sum(pred, dim=1)-pred[:, idx]))/self.sources
        # pred = consistent_pred

        loss, min_idx, parts = self.mixit_loss(
            pred,
            target_waveform,
            return_est=True
        )

        pred_waveform = self.mixit_loss.reorder_source(pred, target_waveform, min_idx, parts)

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, target_waveform).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()

        if not self.current_epoch % 10 and batch_idx == 0 and self.logger is not None:
            preds = pred_waveform[0]
            groundTruths = target_waveform[0]
            label = labels[0]
            i = 0
            table = []
            for waveform_source, waveform_groundTruth, id in zip(preds, groundTruths, label):
                class_label = [id_to_class[item.item()] for item in id]
                table.append([
                    class_label,
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, self.sampling_rate, title=class_label)),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, self.sampling_rate, title=class_label)),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=self.sampling_rate),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=self.sampling_rate),
                ])
                i+=1
            self.logger.log_table(key="results", columns=self.log_columns, data=table)

            isolated = pred[0]
            for idx, source in enumerate(isolated):
                wandb.log({f"Source{idx}": wandb.Audio(source.cpu().numpy(), sample_rate=16000)})


        self.log('val_loss', loss)
        self.log('val_SNR', snr)
        self.log('val_SI-SDR', sdr)

        return loss

    @staticmethod
    def custom_mse(pred, target):
        phase_sensitive_target = torch.abs(target) * torch.cos(torch.angle(target) - torch.angle(pred))
        loss = nn.functional.mse_loss(torch.abs(pred), phase_sensitive_target, reduction="none")
        loss = loss.mean(list(range(1, loss.ndim)))
        return loss
    
    def negative_SNR(self, pred, target):
        # Unsupervised sound separation using mixture invariant training
        soft_threshold = 10**(-30/10)
        loss = 10*torch.log10((target-pred)**2 + soft_threshold*(target**2) + self.epsilon) - 10*torch.log10(target**2 + self.epsilon)
        loss = loss.mean(list(range(1, loss.ndim)))
        return loss
    
    @staticmethod
    def negative_si_sdr(pred, target):
        return -1*scale_invariant_signal_distortion_ratio(pred, target).mean(-1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, maximize=False)
        return optimizer