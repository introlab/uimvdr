import torch
from asteroid.losses import MixITLossWrapper, multisrc_mse
from torch import optim, nn
from torchmetrics.functional import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio import signal_noise_ratio, scale_invariant_signal_distortion_ratio
from torchaudio.transforms import InverseSpectrogram
import pytorch_lightning as pl
import wandb

from ...utils.Windows import cuda_sqrt_hann_window
from ...utils.LabelUtils import id_to_class
from ...utils.PlotUtils import plot_spectrogram_from_waveform
from .ast import ASTModel

class Transformer(pl.LightningModule):
    def __init__(self, in_channels, sources, mics, supervised=True):
        super().__init__()
        self.mics = mics
        self.sources = sources
        self.istft = InverseSpectrogram(
                n_fft=512, hop_length=256, window_fn=cuda_sqrt_hann_window
            )
        # self.mixit_loss = MixITLossWrapper(multisrc_mse if supervised else self.negative_SNR)
        self.mixit_loss = MixITLossWrapper(multisrc_mse)

        self.log_columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.epsilon = torch.finfo(torch.float).eps
        self.supervised = supervised
        output_dim = sources*257*188 if supervised else sources*257*188*2
        self.model = ASTModel(label_dim=output_dim, input_fdim=257, input_tdim=188, model_size='tiny224')
        self.save_hyperparameters()

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        output = self.model(x)
        output = torch.sigmoid(output)
        output = output.reshape(x.shape[0],
                                self.sources if self.supervised else self.sources*2,
                                x.shape[1],
                                x.shape[2])

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
    
        input = torch.abs(mix).median(dim=1, keepdim=True).values

        masks = self(input)

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
    
    def training_step_unsupervised(self, batch, batch_idx):
        """
        Args:
            batch:
                mix 
                isolated sources (batch_num, sources, mics, freq, frame)
        """
        mix, isolatedSources, labels  = batch
    
        mix_of_mix = torch.reshape(mix, (int(mix.shape[0]/2), 2, mix.shape[1], mix.shape[2], mix.shape[3]))

        input_spectrogram = torch.sum(mix_of_mix, dim=1, keepdim=True)
        input = torch.abs(input_spectrogram).median(dim=2, keepdim=False).values

        masks = self(input)

        pred = input_spectrogram[:,:,0] * masks
        # pred_waveform = self.istft(pred)
        # target_waveform = self.istft(mix_of_mix[:, :, 0])
        loss, min_idx, parts = self.mixit_loss(
            torch.abs(pred),
            torch.abs(mix_of_mix[:, :, 0]),
            return_est=True
        )
        pred = self.mixit_loss.reorder_source(pred, mix_of_mix[:, :, 0], min_idx, parts)
        pred_waveform = self.istft(pred)
        target_waveform = self.istft(mix_of_mix[:, :, 0])
        
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
        target = isolatedSources.reshape((-1, self.sources, isolatedSources.shape[2], isolatedSources.shape[3], isolatedSources.shape[4]))
        mix = torch.sum(target, dim=1)

        input = torch.abs(mix).median(dim=1, keepdim=True).values

        masks = self(input)

        pred = mix[:, 0][:, None, ...] * masks

        loss, min_idx, parts = self.mixit_loss(
            torch.abs(pred),
            torch.abs(target[:, :, 0]),
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(pred, target[:, :, 0], min_idx, parts)
        pred_waveform = self.istft(pred)
        target_waveform = self.istft(target[:, :, 0])


        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, target_waveform).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()

        if not self.current_epoch % 49 and batch_idx == 0 and self.logger is not None:
            preds = pred_waveform[0]
            groundTruths = target_waveform[0]
            label = labels[:,0]
            i = 0
            table = []
            for waveform_source, waveform_groundTruth, id in zip(preds, groundTruths, label):
                class_label = id_to_class[id.item()]#[id_to_class[item.item()] for item in id]
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
    
    def validation_step_unsupervised(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        labels = labels.reshape((-1, self.sources, 2))
        # mix isolated sources here to always have the same samples
        target = isolatedSources.reshape((-1, self.sources, isolatedSources.shape[2], isolatedSources.shape[3], isolatedSources.shape[4]))
        mix = torch.sum(target, dim=1)

        mix_of_mix = mix.reshape((int(mix.shape[0]/2), 2, mix.shape[1], mix.shape[2], mix.shape[3]))

        input = torch.abs(torch.sum(mix_of_mix, dim=1, keepdim=True)).median(dim=2, keepdim=False).values

        masks = self(input)

        pred = torch.sum(mix_of_mix[:, :, 0], dim=1, keepdim=True) * masks
        isolated_pred_waveform = self.istft(pred)
        # target_waveform = self.istft(mix_of_mix[:, :, 0])
        # loss, min_idx, parts = self.mixit_loss(
        #     isolated_pred_waveform,
        #     target_waveform,
        #     return_est=True
        # )

        # pred_waveform = self.mixit_loss.reorder_source(isolated_pred_waveform, target_waveform, min_idx, parts)
        loss, min_idx, parts = self.mixit_loss(
            torch.abs(pred),
            torch.abs(mix_of_mix[:, :, 0]),
            return_est=True
        )
        pred = self.mixit_loss.reorder_source(pred, mix_of_mix[:, :, 0], min_idx, parts)
        pred_waveform = self.istft(pred)
        target_waveform = self.istft(mix_of_mix[:, :, 0])

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred_waveform, target_waveform).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()

        if not self.current_epoch % 49 and batch_idx == 0 and self.logger is not None:
            preds = pred_waveform[0]
            groundTruths = target_waveform[0]
            label = labels[0]
            i = 0
            table = []
            for waveform_source, waveform_groundTruth, id in zip(preds, groundTruths, label):
                class_label = [id_to_class[item.item()] for item in id]
                table.append([
                    class_label,
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title=class_label)),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title=class_label)),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                ])
                i+=1
            self.logger.log_table(key="results", columns=self.log_columns, data=table)

            isolated = isolated_pred_waveform[0]
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