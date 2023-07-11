# -*- encoding: utf-8 -*-
'''
@Filename    :model.py
@Time        :2020/07/09 18:22:54
@Author      :Kai Li
@Version     :1.0
'''

import os
from typing import List, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
import wandb
import numpy as np

from ..utils.Windows import cuda_sqrt_hann_window
from ..utils.PlotUtils import plot_spectrogram_from_waveform
from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr, multisrc_mse
from torchmetrics.functional.audio import signal_noise_ratio, scale_invariant_signal_distortion_ratio


class ChannelWiseLayerNorm(pl.LightningModule):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / \
                torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def select_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D_Block(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, dilation=1, norm_type='gLN'):
        super(Conv1D_Block, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, out_channels)
        if norm_type == 'gLN':
            self.padding = (dilation*(kernel_size-1))//2
        else:
            self.padding = dilation*(kernel_size-1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                1, dilation=dilation, padding=self.padding, groups=out_channels, bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, out_channels)
        self.sconv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.norm_type = norm_type

    def forward(self, x):
        w = self.conv1x1(x)
        w = self.norm1(self.prelu1(w))
        w = self.dwconv(w)
        if self.norm_type == 'cLN':
            w = w[:, :, :-self.padding]
        w = self.norm2(self.prelu2(w))
        w = self.sconv(w)
        x = x + w
        return x


class TCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8):
        super(TCN, self).__init__()
        seq = []
        for i in range(X):
            seq.append(Conv1D_Block(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm_type=norm_type, dilation=2**i))
        self.tcn = nn.Sequential(*seq)

    def forward(self, x):
        return self.tcn(x)


class Separation(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8, R=3):
        super(Separation, self).__init__()
        s = [TCN(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=kernel_size, norm_type=norm_type, X=X) for i in range(R)]
        self.sep = nn.Sequential(*s)

    def forward(self, x):
        return self.sep(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, bottleneck=128, kernel_size=16, norm_type='gLN'):
        super(Encoder, self).__init__()
        # self.encoder = nn.Conv1d(
        #     in_channels, out_channels, kernel_size, kernel_size//2, padding=0)
        frame_size = out_channels
        bins = int(frame_size / 2) + 1
        hop_size = int(frame_size / 2)
        self.encoder = torchaudio.transforms.Spectrogram(
            n_fft=frame_size,
            hop_length=hop_size,
            power=None,
            window_fn=cuda_sqrt_hann_window
        )
        self.norm = select_norm(norm_type, bins)
        self.conv1x1 = nn.Conv1d(bins, bottleneck, 1)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        # if x.dim() == 1:
        #     x = torch.unsqueeze(x, 0)
        # if x.dim() == 2:
        #     x = torch.unsqueeze(x, 1)
        x = self.encoder(x)
        w = torch.abs(x)
        w = self.norm(w)
        w = self.conv1x1(w)
        return x, w


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, kernel_size=16):
        super(Decoder, self).__init__()
        # self.decoder = nn.ConvTranspose1d(
        #     in_channels, out_channels, kernel_size, kernel_size//2, padding=0, bias=True)
        frame_size = in_channels
        hop_length = int(frame_size // 2) 
        self.decoder = torchaudio.transforms.InverseSpectrogram(
            n_fft=frame_size, hop_length=hop_length, window_fn=cuda_sqrt_hann_window
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class ConvTasNet(pl.LightningModule):
    def __init__(self,
                 N=1024,
                 L=16,
                 B=128,
                 H=513,
                 P=3,
                 X=8,
                 R=3,
                 norm="gLN",
                 num_spks=2,
                 activate="relu",
                 causal=False,
                 supervised=True
                 ):
        super(ConvTasNet, self).__init__()
        # -----------------------model-----------------------
        self.encoder = Encoder(1, N, B, L, norm)
        self.separation = Separation(B, H, P, norm, X, R)
        self.decoder = Decoder(N, 1, L)
        self.mask = nn.Conv1d(B, H*num_spks, 1, 1)
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if activate not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(activate))
        self.non_linear = supported_nonlinear[activate]
        self.num_spks = num_spks
        self.supervised = supervised
        self.mixit_loss = MixITLossWrapper(self.negative_si_sdr, generalized=False)
        self.epsilon = torch.finfo(torch.float).eps
        self.log_columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]

        self.save_hyperparameters()

    def forward(self, waveform):
        x, w = self.encoder(waveform)
        w = self.separation(w)
        m = self.mask(w)
        # original_shape = m.shape
        # m = torch.reshape(m, (original_shape[0],self.num_spks, original_shape[1]//self.num_spks, original_shape[2]))
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.non_linear(torch.stack(m, dim=0), dim=0)
        # m = self.non_linear(m, dim=1)
        # d = [x*m[:,i] for i in range(self.num_spks)]
        d = [x*m[i] for i in range(self.num_spks)]
        s = torch.empty((waveform.shape[0], self.num_spks, waveform.shape[1]), device=self.device)
        for i in range(self.num_spks):
            s[:, i] = self.decoder(d[i]) 
        
        return s
    
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

        # Ignore mics for now
        mix = mix[:,0]
        isolatedSources = isolatedSources[:,:,0]

        pred = self(mix)

        loss, min_idx, parts = self.mixit_loss(
            pred,
            isolatedSources,
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(pred, isolatedSources, min_idx, parts)
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred, isolatedSources).mean()


        self.log("train_loss", loss, batch_size=mix.shape[0])
        self.log('train_SNR', snr, batch_size=mix.shape[0])
        self.log('train_SI-SDR', sdr, batch_size=mix.shape[0])

        return loss
    
    def validation_step_supervised(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        # Ignore mics for now
        mix = mix[:,0]
        isolatedSources = isolatedSources[:,:,0]

        pred = self(mix)

        loss, min_idx, parts = self.mixit_loss(
            pred,
            isolatedSources,
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(pred, isolatedSources, min_idx, parts)

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred, isolatedSources).mean()

        if not self.current_epoch % 50 and batch_idx == 0 and self.logger is not None:
            preds = pred[1]
            groundTruths = isolatedSources[1]
            label = np.array(labels)
            label = label[:,1]
            i = 0
            table = []
            for waveform_source, waveform_groundTruth, class_label in zip(preds, groundTruths, label):
                table.append([
                    f"{class_label}",
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                ])
                i+=1
            self.logger.log_table(key="results", columns=self.log_columns, data=table)
        
        self.log('val_loss', loss, batch_size=mix.shape[0])
        self.log('val_SNR', snr, batch_size=mix.shape[0])
        self.log('val_SI-SDR', sdr, batch_size=mix.shape[0])

        return loss
    
    def training_step_unsupervised(self, batch, batch_idx):
        mix, isolatedSources, _  = batch

        # Ignore mics for now
        mix = mix[:,0]
        isolatedSources = isolatedSources[:,:,0]

        isolated_mix, mix = self.prepare_data_for_unsupervised(mix)

        isolated_pred = self(mix)

        isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)

        loss, min_idx, parts = self.mixit_loss(
            isolated_pred,
            isolated_mix,
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(isolated_pred, isolated_mix, min_idx, parts)
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolated_mix).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred, isolated_mix).mean()
        
        self.log("train_loss", loss, batch_size=mix.shape[0])
        self.log('train_SNR', snr, batch_size=mix.shape[0])
        self.log('train_SI-SDR', sdr, batch_size=mix.shape[0])

        return loss

    def validation_step_unsupervised(self, batch, batch_idx):
        mix, _, _  = batch

        # Ignore mics for now
        mix = mix[:,0]

        isolated_mix, mix = self.prepare_data_for_unsupervised(mix)

        isolated_pred = self(mix)

        isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)

        loss, min_idx, parts = self.mixit_loss(
            isolated_pred,
            isolated_mix,
            return_est=True
        )

        pred = self.mixit_loss.reorder_source(isolated_pred, isolated_mix, min_idx, parts)
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolated_mix).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred, isolated_mix).mean()
        
        self.log('val_loss', loss, batch_size=mix.shape[0])
        self.log('val_SNR', snr, batch_size=mix.shape[0])
        self.log('val_SI-SDR', sdr, batch_size=mix.shape[0])

        return loss
    
    def validation_epoch_end(self, outputs):
        outputs.clear()

        if not self.current_epoch % 50 and self.logger is not None:
            mix0, isolatedSources0, labels0 = self.trainer.datamodule.dataset_val[0]
            mix1, isolatedSources1, labels1 = self.trainer.datamodule.dataset_val[1]

            mix = torch.stack((mix0, mix1)).to(self.device)
            # isolatedSources = torch.stack((isolatedSources0, isolatedSources1)).to(self.device)
            labels = [labels0, labels1]

            # Ignore mics for now
            mix = mix[:,0]

            isolated_mix, mix = self.prepare_data_for_unsupervised(mix)

            isolated_pred = self(mix)

            isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)

            loss, min_idx, parts = self.mixit_loss(
                isolated_pred,
                isolated_mix,
                return_est=True
            )

            pred = self.mixit_loss.reorder_source(isolated_pred, isolated_mix, min_idx, parts)

            preds = pred[0]
            groundTruths = isolated_mix[0]
            label = np.array(labels).transpose().reshape(-1, self.num_spks)
            label = label[0,:]
            i = 0
            table = []
            for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                table.append([
                    f"{label[i]}, {label[i+2]}",
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                ])
                i+=1
            self.logger.log_table(key="results", columns=self.log_columns, data=table)

            isolated_pred = isolated_pred[0]
            for idx, source in enumerate(isolated_pred):
                wandb.log({f"Source{idx}": wandb.Audio(source.cpu().numpy(), sample_rate=16000)})

        return
    
    def mixture_consistency_projection(self, pred, mix):
        """
            Mixture consistency projection from Mixit article
            Args:
                pred (Tensor): Batch, Sources, Time
                mix (Tensor): Batch, Time

            Returns:
                (Tensor):  Batch, Sources, Time
        """
        consistent_pred = torch.zeros_like(pred)
        for idx in range(self.num_spks):
            consistent_pred[:, idx] = pred[:, idx] + (mix - (torch.sum(pred, dim=1)-pred[:, idx]))/self.num_spks
        return consistent_pred
    
    def prepare_data_for_unsupervised(self, mix):
        """
            Args:
                mix (Tensor): Batch, Time 
            
            Returns:
                (Tuple of tensors): (Batch/2, 2, Time), (Batch/2, Time)
        """
        isolated_mix = mix.reshape(int(mix.shape[0]/2), -1, mix.shape[1])
        new_mix = torch.sum(isolated_mix, dim=1, keepdim=False)

        return isolated_mix, new_mix


    @staticmethod
    def negative_si_sdr(pred, target):
        return -1*scale_invariant_signal_distortion_ratio(pred, target).mean(-1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, maximize=False)
        return optimizer

if __name__ == "__main__":
    conv = ConvTasNet()
    a = torch.randn(4, 4096)
    s = conv(a)
