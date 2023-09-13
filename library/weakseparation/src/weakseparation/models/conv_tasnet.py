# https://github.com/JusperLee/Conv-TasNet
# -*- encoding: utf-8 -*-
'''
@Filename    :model.py
@Time        :2020/07/09 18:22:54
@Author      :Kai Li
@Version     :1.0
'''

import os
from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
import wandb
import numpy as np
import random

from ..utils.Windows import cuda_sqrt_hann_window
from ..utils.PlotUtils import plot_spectrogram_from_waveform
from .efficient_net import EffNetAttention
from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr, multisrc_mse
from torchmetrics.functional.audio import signal_noise_ratio, scale_invariant_signal_distortion_ratio
from pytorch_lightning.loggers import WandbLogger


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
        self.hop_length = int(frame_size // 2) 
        self.decoder = torchaudio.transforms.InverseSpectrogram(
            n_fft=frame_size, hop_length=self.hop_length, window_fn=cuda_sqrt_hann_window
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
                 classi_percent=0.5,
                 alpha=1,
                 beta=1,
                 gamma=0.5,
                 norm="gLN",
                 num_spks=2,
                 activate="relu",
                 causal=False,
                 supervised=True,
                 learning_rate=1e-4
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
        self.learning_rate = learning_rate
        self.epsilon = torch.finfo(torch.float).eps
        self.log_columns = ["class", "Si-SDR", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.gamma = gamma
        self.beta = beta

        # Classification
        self.classi_percentage = classi_percent
        if classi_percent > 0:
            sd = torch.load('/home/jacob/dev/weakseparation/library/weakseparation/src/best_models/fsd_mdl_wa.pth', map_location=self.device)
            self.classi_model = EffNetAttention(label_dim=200, b=2, pretrain=False, head_num=4)
            self.classi_model = torch.nn.DataParallel(self.classi_model)
            self.classi_model.load_state_dict(sd, strict=False)
            self.classi_model.eval()
            self.classi_epsilon = 1e-7
            self.classi_audio_conf = {'num_mel_bins': 128, 'target_length': 3000, 'freqm': 0, 'timem': 0, 'mixup': 0,
                                    'dataset': "FSD50K", 'mode': 'evaluation', 'mean': -4.6476,
                                    'std': 4.5699, 'noise': False}
            self.classi_loss_fnct = torch.nn.BCELoss()
            self.alpha = alpha

        self.save_hyperparameters()

    def forward(self, waveform, return_target_spectrogram=False):
        x, w = self.encoder(waveform)
        w = self.separation(w)
        m = self.mask(w)
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.non_linear(torch.stack(m, dim=0), dim=0)
        d = [x*m[i] for i in range(self.num_spks)]
        s = torch.empty((waveform.shape[0], self.num_spks, waveform.shape[1]), device=self.device)
        for i in range(self.num_spks):
            pred = self.decoder(d[i])
            pred = torch.nn.functional.pad(pred, (0, int(waveform.shape[1]-pred.shape[1])), mode="constant", value=0)
            s[:, i] = pred
        if not return_target_spectrogram:
            return s
        else: 
            return s, torch.abs(x)*m[0]
    
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

        isolatedSources = self.prepare_data_for_supervised(isolatedSources)

        pred = self(mix)

        # loss, pred = self.compute_loss(pred, isolatedSources)
        pred = self.efficient_mixit(pred, isolatedSources)
        
        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred, isolatedSources).mean()
        loss = -1 * sdr


        self.log("train_loss", loss, batch_size=mix.shape[0])
        self.log('train_SNR', snr, batch_size=mix.shape[0])
        self.log('train_SI_SDR', sdr, batch_size=mix.shape[0])

        return loss
    
    def validation_step_supervised(self, batch, batch_idx):
        mix, isolatedSources, labels  = batch

        # Ignore mics for now
        mix = mix[:,0]
        isolatedSources = isolatedSources[:,:,0]

        isolatedSources = self.prepare_data_for_supervised(isolatedSources)

        pred = self(mix)

        # loss, pred = self.compute_loss(pred, isolatedSources)
        pred = self.efficient_mixit(pred, isolatedSources)
        target_si_sdr = scale_invariant_signal_distortion_ratio(pred[:,0], isolatedSources[:,0]).mean()

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolatedSources).mean()
        sdr = scale_invariant_signal_distortion_ratio(pred, isolatedSources).mean()
        loss = -1 * sdr
        
        self.log('val_loss', loss, batch_size=mix.shape[0])
        self.log('val_SNR', snr, batch_size=mix.shape[0])
        self.log('val_SI_SDR', sdr, batch_size=mix.shape[0])
        self.log('val_target_SI_SDR', target_si_sdr, batch_size=mix.shape[0])

        return loss
    
    def training_step_unsupervised(self, batch, batch_idx):
        mix, isolated_sources, _, classi_labels = batch

        # Ignore mics for now
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        isolated_mix, mix = self.prepare_data_for_unsupervised(isolated_sources)

        isolated_pred, target_pred_spectrogram = self(mix, return_target_spectrogram=True)

        isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)

        # pred = self.efficient_mixit(isolated_pred, isolated_mix)

        least_squares_result = torch.linalg.lstsq(isolated_pred.transpose(1,2), isolated_mix.transpose(1,2)).solution.transpose(1,2)

        max_indices = torch.argmax(least_squares_result, dim=1)

        # Create a binary mask where 1 is set at the indices of minimum values, rest to 0
        mask = torch.zeros_like(least_squares_result)
        mask.scatter_(1, max_indices.unsqueeze(1), 1)

        # Forcing target
        mask[..., 0, 0] = 1
        mask[..., 1, 0] = 0

        pred = torch.bmm(mask, isolated_pred)

        # ground_truth_permutation = torch.zeros_like(mask[..., 0])
        # # Target should always be in first mixture
        # ground_truth_permutation[..., 0] = 1

        # permutation_loss = self.delta * self.classi_loss_fnct(mask[..., 0], ground_truth_permutation)

        loss = -1*scale_invariant_signal_distortion_ratio(pred, isolated_mix).mean()

        # Classification
        classi_loss = 0.0
        if random.random()<self.classi_percentage:
            # Need to manually clear classification model grad because not in optimzer and lightning calls optimizer.grad
            self.classi_model.zero_grad()

            #TODO: Make it functional for more than the first 2 when training on bigger GPU
            target_pred = isolated_pred[:2].transpose(0,1)
            for idx, source in enumerate(target_pred):
                fbank_target = torch.zeros(source.shape[0], self.classi_audio_conf["target_length"], self.classi_audio_conf["num_mel_bins"])
                for i, waveform in enumerate(source):
                    fbank_target[i] = self.wav2fbank(waveform[None, ...])
                prediction = self.classi_model(fbank_target)
                prediction = torch.clamp(prediction, self.classi_epsilon, 1. - self.classi_epsilon)

                if idx>=1:
                    # TODO: pass ignore index as parameters
                    # Just to test for now: Bark and Dog
                    indices = torch.tensor([7,59], dtype=torch.long)
                    prediction = prediction[:, indices]
                    label_ground_truth = torch.zeros_like(prediction, dtype=torch.float32)
                    classi_loss += self.classi_loss_fnct(prediction, label_ground_truth)
                else:
                    classi_loss += self.classi_loss_fnct(prediction, classi_labels[:2])

            classi_loss = self.alpha*classi_loss
            self.log('train_classi_loss', classi_loss)

        target_energy = self.gamma * torch.mean((target_pred_spectrogram+self.epsilon)**self.beta)
        loss = loss + classi_loss + target_energy
        
        with torch.no_grad():
            # Pytorch si-snr is si-sdr
            snr = signal_noise_ratio(pred, isolated_mix).mean()
            sdr = scale_invariant_signal_distortion_ratio(pred, isolated_mix).mean()
        
            self.log('train_SNR', snr, batch_size=mix.shape[0])
            self.log('train_SI_SDR', sdr, batch_size=mix.shape[0])
            self.log("train_loss", loss, batch_size=mix.shape[0])

        return loss

    def validation_step_unsupervised(self, batch, batch_idx):
        mix, isolated_sources, labels, _  = batch

        # Ignore mics for now
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        isolated_mix, mix = self.prepare_data_for_unsupervised(isolated_sources)

        isolated_pred = self(mix)

        isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)

        pred = self.efficient_mixit(isolated_pred, isolated_mix)
        target_si_sdr = scale_invariant_signal_distortion_ratio(isolated_pred[:,0], isolated_sources[:,0]).mean()
        isolated_pred, permutation_loss = self.efficient_mixit(isolated_pred, isolated_sources, True)
        permutation_loss = torch.mean(permutation_loss[..., 1:])
        loss = -1*scale_invariant_signal_distortion_ratio(pred, isolated_mix).mean()
        # loss, pred = self.compute_loss(isolated_pred, isolated_mix)
        # _, isolated_pred = self.compute_loss(isolated_pred, isolated_sources)

        # Pytorch si-snr is si-sdr
        snr = signal_noise_ratio(pred, isolated_mix).mean()
        
        msi = scale_invariant_signal_distortion_ratio(isolated_pred, isolated_sources)
        # Get mean without the empty sources (like in Mixit original paper)
        msi_count = 0
        msi_nb = 0
        for i, label in enumerate(labels):
            for j, l_batch in enumerate(label):
                if l_batch != "nothing":
                    msi_count += msi[j, i]
                    msi_nb +=1
        msi = msi_count/msi_nb

        sdr = scale_invariant_signal_distortion_ratio(pred, isolated_mix).mean()

        self.log('val_permutation_loss', permutation_loss, batch_size=mix.shape[0])
        self.log('val_loss', loss, batch_size=mix.shape[0])
        self.log('val_SNR', snr, batch_size=mix.shape[0])
        self.log('val_SI_SDR', sdr, batch_size=mix.shape[0])
        self.log('val_MSi', msi, batch_size=mix.shape[0])
        self.log('val_target_SI_SDR', target_si_sdr, batch_size=mix.shape[0])

        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        mix, isolated_sources, labels = batch

        # Ignore mics for now
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        if not self.supervised:
            target, mix = self.prepare_data_for_unsupervised(isolated_sources)
        else:
            target = self.prepare_data_for_supervised(isolated_sources)

        isolated_pred, target_pred_mask = self(mix, return_target_spectrogram=True)

        if not self.supervised:
            isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)

        pred = self.efficient_mixit(isolated_pred, target)

        sdr = scale_invariant_signal_distortion_ratio(pred, target).mean()

        if not self.supervised:
            target_si_sdr = scale_invariant_signal_distortion_ratio(isolated_pred[:,0], isolated_sources[:,0]).mean()
            _, orig_permutation_loss = self.efficient_mixit(isolated_pred, isolated_sources, True)
            permutation_loss = torch.mean(orig_permutation_loss[..., 1:])

            self.log_dict({
                "test_si_sdr" : sdr,
                "test_target_si_sdr" : target_si_sdr,
                "permutation_loss" : permutation_loss,
            }, batch_size=mix.shape[0])
        else:
            isolated_pred = self.efficient_mixit(isolated_pred, isolated_sources)
            target_si_sdr = scale_invariant_signal_distortion_ratio(isolated_pred[:,0], isolated_sources[:,0]).mean()

            self.log_dict({
                "test_si_sdr" : sdr,
                "test_target_si_sdr" : target_si_sdr,
            }, batch_size=mix.shape[0])

    def on_test_end(self):
        if isinstance(self.logger, WandbLogger):
            mix0, isolatedSources0, labels0 = self.trainer.datamodule.dataset_test_respeaker.get_serialized_sample(15, 1300)
            mix1, isolatedSources1, labels1 = self.trainer.datamodule.dataset_test_respeaker.get_serialized_sample(10, 1300)

            mix = torch.stack((mix0, mix1)).to(self.device)
            isolatedSources = torch.stack((isolatedSources0, isolatedSources1)).to(self.device)
            labels = [labels0, labels1]

            # Ignore mics for now
            mix = mix[:,0]
            isolatedSources = isolatedSources[:,:,0]

            if not self.supervised:
                target, mix = self.prepare_data_for_unsupervised(isolatedSources)
            else:
                target = self.prepare_data_for_supervised(isolatedSources)

            isolated_pred = self(mix)

            if not self.supervised:
                isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)
                isolated_pred = self.efficient_mixit(isolated_pred, isolatedSources)

            pred = self.efficient_mixit(isolated_pred, target)

            preds = pred[0]
            groundTruths = target[0]
            label = labels[0]
            i = 0
            table = []

            if not self.supervised:
                for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                    table.append([
                        f"{label[i]}, {label[i+1]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean(),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=2
                self.logger.log_table(key="results", columns=self.log_columns, data=table)

                isolated_pred = isolated_pred[0]
                isolated_sources = isolatedSources[0]
                i = 0
                table = []
                for waveform_source, waveform_groundTruth in zip(isolated_pred, isolated_sources):
                    table.append([
                        f"{label[i]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean(),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=1
                self.logger.log_table(key="isolated results", columns=self.log_columns, data=table)

            wandb.log({f"Mix": wandb.Audio(mix[0].cpu().numpy(), sample_rate=16000)})

        return

    
    def validation_epoch_end(self, outputs):
        outputs.clear()

        if not self.current_epoch % 100 and isinstance(self.logger, WandbLogger):
            mix0, isolatedSources0, labels0 = self.trainer.datamodule.dataset_val.get_serialized_sample(15, 1300)
            mix1, isolatedSources1, labels1 = self.trainer.datamodule.dataset_val.get_serialized_sample(10, 1300)

            mix = torch.stack((mix0, mix1)).to(self.device)
            isolatedSources = torch.stack((isolatedSources0, isolatedSources1)).to(self.device)
            labels = [labels0, labels1]

            # Ignore mics for now
            mix = mix[:,0]
            isolatedSources = isolatedSources[:,:,0]

            if not self.supervised:
                target, mix = self.prepare_data_for_unsupervised(isolatedSources)
            else:
                target = self.prepare_data_for_supervised(isolatedSources)

            isolated_pred = self(mix)

            if not self.supervised:
                isolated_pred = self.mixture_consistency_projection(isolated_pred, mix)
                isolated_pred = self.efficient_mixit(isolated_pred, isolatedSources)

            pred = self.efficient_mixit(isolated_pred, target)

            preds = pred[0]
            groundTruths = target[0]
            label = labels[0]
            i = 0
            table = []

            if not self.supervised:
                for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                    table.append([
                        f"{label[i]}, {label[i+1]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean(),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=2
                self.logger.log_table(key="results", columns=self.log_columns, data=table)

                isolated_pred = isolated_pred[0]
                isolatedSources = isolatedSources[0]
                i = 0
                table = []
                for waveform_source, waveform_groundTruth in zip(isolated_pred, isolatedSources):
                    table.append([
                        f"{label[i]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean(),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=1
                self.logger.log_table(key="isolated results", columns=self.log_columns, data=table)
            else:
                new_label: list = []
                new_label.append(label[0])

                new_label.append(' '.join(map(str,label[1:])))

                label = new_label
                for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                    table.append([
                        f"{label[i]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean(),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=1
                self.logger.log_table(key="results", columns=self.log_columns, data=table)

            wandb.log({f"Mix": wandb.Audio(mix[0].cpu().numpy(), sample_rate=16000)})

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

    def prepare_data_for_unsupervised(self, isolated_sources):
        """
            Args:
                isolated_sources (Tensor): Batch, Sources, Time 
            
            Returns:
                (Tuple of tensors): (Batch, Sources/2, 2, Time), (Batch, Time)
        """
        isolated_mix = isolated_sources.reshape(isolated_sources.shape[0], -1, 2, isolated_sources.shape[2])
        isolated_mix = torch.sum(isolated_mix, dim=2, keepdim=False)
        new_mix = torch.sum(isolated_sources, dim=1, keepdim=False)

        return isolated_mix, new_mix
    
    def prepare_data_for_supervised(self, isolated_sources):
        """
            Args:
                isolated_sources (Tensor): Target signal needs to be the first source (Batch, Sources, Time)
            
            Returns:
                (Tensor): (Batch, 2, Time)
        """

        additionnal_sources_mix = isolated_sources[:, 1:, ...]

        additionnal_sources_mix = torch.sum(additionnal_sources_mix, 1)

        new_isolated_source = torch.empty(isolated_sources.shape[0], 2, isolated_sources.shape[2], device=self.device)

        new_isolated_source[:,0] = isolated_sources[:,0]
        new_isolated_source[:,1] = additionnal_sources_mix

        return new_isolated_source
    
    def compute_loss(self, pred, target):
        loss, min_idx, parts = self.mixit_loss(
            pred,
            target,
            return_est=True
        )

        new_pred = self.mixit_loss.reorder_source(pred, target, min_idx, parts)

        return loss, new_pred
    
    def wav2fbank(self, wave):
        waveform = wave - wave.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.classi_audio_conf["num_mel_bins"], dither=0.0, frame_shift=10)

        target_length = self.classi_audio_conf["target_length"]
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        #Normalization
        fbank = (fbank - self.classi_audio_conf["mean"]) / self.classi_audio_conf["std"]

        return fbank

    @staticmethod
    def negative_si_sdr(pred, target):
        return -1*scale_invariant_signal_distortion_ratio(pred, target).mean(-1)
    
    @staticmethod 
    def efficient_mixit(pred, target, return_target_permutation = False):
        least_squares_result = torch.linalg.lstsq(pred.transpose(1,2), target.transpose(1,2)).solution.transpose(1,2)

        max_indices = torch.argmax(least_squares_result, dim=1)

        # Create a binary mask where 1 is set at the indices of minimum values, rest to 0
        mask = torch.zeros_like(least_squares_result)
        mask.scatter_(1, max_indices.unsqueeze(1), 1)

        mixit_result = torch.bmm(mask, pred)

        if not return_target_permutation:
            return mixit_result
        else:
            return mixit_result, mask[..., 0]
    
    def configure_optimizers(self):
        params_to_optimize = [param for name, param in self.named_parameters() if name.split(".")[0] != "classi_model"]
        optimizer = optim.Adam(params_to_optimize, lr=self.learning_rate, maximize=False)
        return optimizer

if __name__ == "__main__":
    conv = ConvTasNet()
    a = torch.randn(4, 4096)
    s = conv(a)
