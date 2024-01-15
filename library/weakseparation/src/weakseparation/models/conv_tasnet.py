# https://github.com/JusperLee/Conv-TasNet
# -*- encoding: utf-8 -*-
'''
@Filename    :model.py
@Time        :2020/07/09 18:22:54
@Author      :Kai Li
@Version     :1.0
'''

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
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from pytorch_lightning.loggers import WandbLogger


class ChannelWiseLayerNorm(nn.LayerNorm):
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
    if norm not in ["cLN", "gLN", "BN", "fLN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    elif norm == "fLN":
        return nn.InstanceNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D_Block(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, dilation=1, norm_type='gLN', layer=0):
        super(Conv1D_Block, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.scale1 = nn.Parameter(torch.tensor([1.0]))
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, out_channels)
        if norm_type == 'gLN' or norm_type == 'fLN':
            self.padding = (dilation*(kernel_size-1))//2
        else:
            self.padding = dilation*(kernel_size-1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                1, dilation=dilation, padding=self.padding, groups=out_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, out_channels)
        self.sconv = nn.Conv1d(out_channels, in_channels, 1)
        self.scale2 = nn.Parameter(torch.tensor([0.9**layer]))
        self.skip_connection = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x):
        w = self.conv1x1(x) * self.scale1
        w = self.norm1(self.prelu1(w))
        w = self.dwconv(w)
        w = self.norm2(self.prelu2(w))
        skip = self.skip_connection(w)

        out = x + (self.sconv(w) * self.scale2)
        
        return out, skip


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
        x = self.encoder(x)
        w = torch.abs(x)
        w = self.norm(w)
        w = self.conv1x1(w)
        return x, w


class Decoder(nn.Module):
    def __init__(self, in_channels=512):
        super(Decoder, self).__init__()
        frame_size = in_channels
        self.hop_length = int(frame_size // 2) 
        self.decoder = torchaudio.transforms.InverseSpectrogram(
            n_fft=frame_size, hop_length=self.hop_length, window_fn=cuda_sqrt_hann_window
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class ConvTasNet(pl.LightningModule):
    """
        TCDN++ adaptation of the ConvTasNet model
        Args:
            N (int) : Frame size
            B (int) : Bottleneck size
            H (int) : Output size of 1D convs
            P (int) : Kernel size of 1D convs
            X (int) : Number of 1D convs in one pass
            R (int) : Number of repeats
            beta (float) : Power of the energy
            gamma (float) : Weight of the energy
            kappa (float) : Weight of sparsity loss
            norm (str) : Type of normalization 
            num_spks (int) : number of outputs of the network
            activate (str) : type of activation layer
            supervised (bool) : Whether to train supervised or unsupervised
            learning rate (float) : learning rate for the training
    """
    def __init__(self,
                 N=1024,
                 B=256,
                 H=512,
                 P=3,
                 X=8,
                 R=4,
                 beta=1,
                 gamma=0.0,
                 kappa=0.0,
                 norm="gLN",
                 num_spks=2,
                 activate="softmax",
                 supervised=True,
                 learning_rate=1e-4
                 ):
        super(ConvTasNet, self).__init__()
        # -----------------------model-----------------------
        self.encoder = Encoder(1, N, B, norm)
        self.separation = self._Sequential_repeat(
            R, X, in_channels=B, out_channels=H, kernel_size=P, norm_type=norm
        )
        self.decoder = Decoder(N)
        self.final_prelu = nn.PReLU()
        bins = int(N / 2) + 1
        self.mask = nn.Conv1d(B, bins*num_spks, 1, 1)
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
        self.learning_rate = learning_rate
        self.epsilon = torch.finfo(torch.float).eps
        self.log_columns = ["class", "Si-SDRi", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.num_blocks = X
        self.index_of_last_skip = R*X - X
        nb_of_longterm_skip = int(self.index_of_last_skip/X)
        longterm_skip_lists = []
        for i in range(nb_of_longterm_skip):
            longterm_skip_lists += [nn.Conv1d(B, B, 1)]
        self.longterm_skip = nn.ModuleList(longterm_skip_lists)

        # Beamforming
        self.scm_transform = torchaudio.transforms.PSD()
        self.mvdr_transform = torchaudio.transforms.SoudenMVDR()

        self.save_hyperparameters()

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           args:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return Conv1D_Block_lists

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           args:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = []
        for i in range(num_repeats):
            repeats_lists += self._Sequential_block(
                num_blocks, **block_kwargs)
        return nn.ModuleList(repeats_lists)

    def forward(self, waveform, return_pred_spectrogram=False, return_pred_mask=False):
        """
            Forward pass of model, this always returns the predicted waveforms but you can enable 
            the flags to retrun spectrograms or masks.

            Args:
                waveform (Tensor) : Waveform to separate (batch, time)
                return_pred_spectrogram (bool) : Return predicted spectrogram without decoding
                return_pred_mask (bool) : Return predicted mask
            
            Return:
                waveform (Tensor) : Sepated waveform (batch, sources, time)
                spectrogram (Tensor) : Sepated waveform (batch, sources, freq, frame)
                mask (Tensor) : Predicted mask between 0 and 1 (batch, sources, freq, frame)
        """
        # Encoding
        x, w = self.encoder(waveform)

        # Separation w/ skip connections
        short_term_skip_connection = torch.zeros((w.shape[0], w.shape[1], w.shape[2]), device=w.device)
        long_term_skip_connections = []
        for i in range(len(self.separation)):
            if not i%self.num_blocks and i <= self.index_of_last_skip:
                for connection in long_term_skip_connections:
                    w += connection
                long_term_index = int(i/self.num_blocks)
                if i < self.index_of_last_skip:
                    long_term_skip_connections.append(self.longterm_skip[long_term_index](w))

            w, skip = self.separation[i](w)
            short_term_skip_connection += skip
        w = short_term_skip_connection

        w = self.final_prelu(w)
        m = self.mask(w)
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.non_linear(torch.stack(m, dim=0), dim=0)

        # Apply masks
        d = [x*m[i] for i in range(self.num_spks)]

        # Decoding
        s = torch.empty((waveform.shape[0], self.num_spks, waveform.shape[1]), device=self.device)
        for i in range(self.num_spks):
            pred = self.decoder(d[i])
            pred = torch.nn.functional.pad(pred, (0, int(waveform.shape[1]-pred.shape[1])), mode="constant", value=0)
            s[:, i] = pred

        s = self.mixture_consistency_projection(s, waveform)

        if return_pred_spectrogram and not return_pred_mask:
            return s, torch.abs(x)*m[0]
        elif not return_pred_spectrogram and return_pred_mask: 
            return s, m[0]
        elif return_pred_spectrogram and return_pred_mask: 
            return s, torch.abs(x)*m[0], m[0]
        else:
            return s
    
    def training_step(self, batch, batch_idx):
        """
            Pytorch lighning function called for training

            Args:
                batch (Tensor) : batch to process for training
                batch_idx (int) : Number of batch in epoch
            
            Return:
                (float) : Training loss for backward propagation and updating weights
        """
        if self.supervised:
            loss = self.training_step_supervised(batch, batch_idx)
        else:
            loss = self.training_step_unsupervised(batch, batch_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
            Pytorch lighning function called for validation

            Args:
                batch (Tensor) : batch to process for validation
                batch_idx (int) : Current batch number in epoch
            
            Return:
                (float) : Validation loss
        """
        if self.supervised:
            loss = self.validation_step_supervised(batch, batch_idx)
        else:
            loss = self.validation_step_unsupervised(batch, batch_idx)

        return loss
    
    def training_step_supervised(self, batch, batch_idx):
        """
        Supervised training

        Args:
            batch (Tuple):
                mix (batch_num, mics, freq, frame)
                isolated sources (batch_num, sources, mics, freq, frame)
                labels (sources, batch)
            batch_idx (int): Current batch number in epoch

        Return:
            (float) : Training loss for backward propagation and updating weights
        """
        mix, isolated_sources, labels  = batch

        # Ignore mics for model prediction
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        isolated_sources, labels = self.prepare_data_for_supervised(isolated_sources, labels)

        pred = self(mix)

        pred = self.efficient_mixit(pred, isolated_sources, force_target=True)
        
        loss = self.negative_SNR(pred, isolated_sources)

        msi = self.compute_msi(pred, isolated_sources, mix, labels)

        self.log_dict({
            "train_loss": loss,
            "train_MSi": msi,
        }, batch_size=mix.shape[0], sync_dist=True)

        return loss
    
    def validation_step_supervised(self, batch, batch_idx):
        """
        Supervised validation

        Args:
            batch (Tuple):
                mix (batch_num, mics, freq, frame)
                isolated sources (batch_num, sources, mics, freq, frame)
                labels (sources, batch)
            batch_idx (int): Current batch number in epoch

        Return:
            (float) : Validation loss
        """
        mix, isolated_sources, labels  = batch

        # Ignore mics for model prediction
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        isolated_sources, labels = self.prepare_data_for_supervised(isolated_sources, labels)

        pred = self(mix)

        pred = self.efficient_mixit(pred, isolated_sources, force_target=True)

        loss = self.negative_SNR(pred, isolated_sources)

        # Target SI_SDRi
        target_si_sdr = scale_invariant_signal_distortion_ratio(pred[:,0], isolated_sources[:,0]).mean()
        target_si_sdri = target_si_sdr - scale_invariant_signal_distortion_ratio(mix, isolated_sources[:,0]).mean()
        
        msi = self.compute_msi(pred, isolated_sources, mix, labels)
        
        self.log_dict({
            'val_loss': loss,
            'val_MSi': msi,
            'val_target_SI_SDRi': target_si_sdri,
        }, batch_size=mix.shape[0], sync_dist=True)

        return loss
    
    def training_step_unsupervised(self, batch, batch_idx):
        """
        Unsupervised training

        Args:
            batch (Tuple):
                mix (batch_num, mics, freq, frame)
                isolated sources (batch_num, sources, mics, freq, frame)
                labels (sources, batch)
            batch_idx (int): Current batch number in epoch

        Return:
            (float) : Training loss for backward propagation and updating weights
        """
        mix, isolated_sources, _ = batch

        # Ignore mics for model prediction
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        isolated_mix, isolated_sources, _ = self.prepare_data_for_unsupervised(isolated_sources)

        isolated_pred, target_pred_spectrogram = self(mix, return_pred_spectrogram=True)

        # Efficient mixit with forcing target
        pred = self.efficient_mixit(isolated_pred, isolated_mix, force_target=True)

        loss = self.negative_SNR(pred, isolated_mix) 

        sparcity_loss = 0.0
        if self.kappa:
            rms = torch.square(torch.mean(isolated_pred**2, dim=-1))
            sparcity_loss = self.kappa * rms.mean()/(rms**2).sum()

        target_energy = 0.0
        if self.gamma:
            target_energy = self.gamma * torch.mean((target_pred_spectrogram+self.epsilon)**self.beta)

        loss = loss + target_energy + sparcity_loss
        
        with torch.no_grad():
            repeated_mix = mix[:, None, :]
            repeated_mix = repeated_mix.expand(-1, isolated_mix.shape[1], -1)
            si_sdri = (scale_invariant_signal_distortion_ratio(pred, isolated_mix) - scale_invariant_signal_distortion_ratio(repeated_mix, isolated_mix)).mean()
        
            self.log_dict({
                "train_MoMi": si_sdri,
                "train_loss": loss,
            }, batch_size=mix.shape[0], sync_dist=True)

        return loss

    def validation_step_unsupervised(self, batch, batch_idx):
        """
        Unsupervised validation

        Args:
            batch (Tuple):
                mix (batch_num, mics, freq, frame)
                isolated sources (batch_num, sources, mics, freq, frame)
                labels (sources, batch)
            batch_idx (int): Current batch number in epoch

        Return:
            (float) : Validation loss
        """
        mix, isolated_sources, labels  = batch

        # Ignore mics for model prediction
        mix = mix[:,0]
        isolated_sources = isolated_sources[:,:,0]

        isolated_mix, isolated_sources, labels = self.prepare_data_for_unsupervised(isolated_sources, labels)

        isolated_pred = self(mix)

        pred = self.efficient_mixit(isolated_pred, isolated_mix, force_target=True)

        loss = self.negative_SNR(pred, isolated_mix)

        # Target SI_SDRi
        target_si_sdr = scale_invariant_signal_distortion_ratio(isolated_pred[:,0], isolated_sources[:,0]).mean()
        target_si_sdri = target_si_sdr - scale_invariant_signal_distortion_ratio(mix, isolated_sources[:,0]).mean()

        isolated_pred = self.efficient_mixit(isolated_pred, isolated_sources, force_target=True)
        
        msi = self.compute_msi(isolated_pred, isolated_sources, mix, labels)

        repeated_mix = mix[:, None, :]
        repeated_mix = repeated_mix.expand(-1, isolated_mix.shape[1], -1)
        si_sdri = (scale_invariant_signal_distortion_ratio(pred, isolated_mix) - scale_invariant_signal_distortion_ratio(repeated_mix, isolated_mix)).mean()

        self.log_dict({
            'val_loss': loss,
            'val_MoMi': si_sdri,
            'val_MSi': msi,
            'val_target_SI_SDRi': target_si_sdri,
        }, batch_size=mix.shape[0], sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Test step, this is where the beamforming is applied.

        Args:
            batch (Tuple):
                mix (batch_num, mics, freq, frame)
                isolated sources (batch_num, sources, mics, freq, frame)
                labels (sources, batch)
            batch_idx (int): Current batch number in epoch
            dataloader_idx (int): Current dataloader number (There's multiple test datasets)
        """
        multimic_mix, multimic_isolated_sources, orig_labels = batch

        # Ignore mics for now
        mix = multimic_mix[:,0]
        isolated_sources = multimic_isolated_sources[:,:,0]

        if not self.supervised:
            MoM, isolated_sources, labels = self.prepare_data_for_unsupervised(isolated_sources, orig_labels)
        else:
            isolated_sources, labels = self.prepare_data_for_supervised(isolated_sources, orig_labels)

        isolated_pred, pred_mask = self(mix, return_pred_mask=True)

        target_si_sdr = scale_invariant_signal_distortion_ratio(isolated_pred[:,0], isolated_sources[:,0]).mean()
        mix_si_sdr = scale_invariant_signal_distortion_ratio(mix, isolated_sources[:,0]).mean()
        target_si_sdri = target_si_sdr - mix_si_sdr

        stft_mix = self.encoder.encoder(multimic_mix)
        stft_target = self.encoder.encoder(multimic_isolated_sources[:,0])
        oracle_mask = torch.abs(stft_target[:,0]) / (torch.abs(stft_target[:,0]) + torch.abs(stft_mix[:,0]-stft_target[:,0]) + self.epsilon)
        stft_oracle_pred = (stft_mix.transpose(0,1) * oracle_mask).transpose(0,1)

        oracle_pred = self.decoder.decoder(stft_oracle_pred[:,0])
        oracle_pred = torch.nn.functional.pad(oracle_pred, (0, int(mix.shape[-1]-oracle_pred.shape[-1])), mode="constant", value=0)

        oracle_si_sdr = scale_invariant_signal_distortion_ratio(oracle_pred, isolated_sources[:,0]).mean()
        oracle_si_sdri = oracle_si_sdr - mix_si_sdr

        speech = self.trainer.datamodule.dataset_test_16sounds.target_class == "Speech"
        if speech:
            try:
                pesq = perceptual_evaluation_speech_quality(isolated_pred[:,0], isolated_sources[:,0], 16000, 'wb').mean()
            except:
                pesq = None
            stoi = short_time_objective_intelligibility(isolated_pred[:,0], isolated_sources[:,0], 16000, ).mean()

        if multimic_mix.shape[1] > 1:
            # Get Spectrograms
            stft_pred = (stft_mix.transpose(0,1) * pred_mask).transpose(0,1)
            stft_noise = stft_mix - stft_pred
            stft_oracle_noise = stft_mix - stft_oracle_pred

            # Get Spatial covariance matrices
            scm_pred = self.scm_transform(stft_pred)
            scm_pred_noise = self.scm_transform(stft_noise)
            scm_oracle = self.scm_transform(stft_oracle_pred)
            scm_oracle_noise = self.scm_transform(stft_oracle_noise)

            # Compute beamforming weight and apply it
            beamformed_pred = self.mvdr_transform(stft_mix, scm_pred, scm_pred_noise, reference_channel=0)
            beamformed_target = self.mvdr_transform(stft_target, scm_pred, scm_pred_noise, reference_channel=0)
            beamformed_oracle = self.mvdr_transform(stft_mix, scm_oracle, scm_oracle_noise, reference_channel=0)
            beamformed_oracle_target = self.mvdr_transform(stft_target, scm_oracle, scm_oracle_noise, reference_channel=0)

            # Post mask
            post_mask = pred_mask.clone()
            post_mask[post_mask<0.3] = 0.3
            beamformed_pred = beamformed_pred*pred_mask

            # Spectrogram -> Waveform
            beamformed_pred = self.decoder.decoder(beamformed_pred)
            beamformed_pred = torch.nn.functional.pad(beamformed_pred, (0, int(mix.shape[-1]-beamformed_pred.shape[-1])), mode="constant", value=0)
            beamformed_target = self.decoder.decoder(beamformed_target)
            beamformed_target = torch.nn.functional.pad(beamformed_target, (0, int(mix.shape[-1]-beamformed_target.shape[-1])), mode="constant", value=0)
            beamformed_oracle = self.decoder.decoder(beamformed_oracle)
            beamformed_oracle = torch.nn.functional.pad(beamformed_oracle, (0, int(mix.shape[-1]-beamformed_oracle.shape[-1])), mode="constant", value=0)
            beamformed_oracle_target = self.decoder.decoder(beamformed_oracle_target)
            beamformed_oracle_target = torch.nn.functional.pad(beamformed_oracle_target, (0, int(mix.shape[-1]-beamformed_oracle_target.shape[-1])), mode="constant", value=0)

            # Compute metrics
            beam_target_si_sdr = scale_invariant_signal_distortion_ratio(beamformed_pred, beamformed_target).mean()
            beam_target_si_sdri = beam_target_si_sdr - mix_si_sdr

            beam_oracle_si_sdr = scale_invariant_signal_distortion_ratio(beamformed_oracle, beamformed_oracle_target).mean()
            beam_oracle_si_sdri = beam_oracle_si_sdr - mix_si_sdr

            if speech:
                try:
                    beam_pesq = perceptual_evaluation_speech_quality(beamformed_pred, beamformed_target, 16000, 'wb').mean()
                except:
                    beam_pesq = None
                beam_stoi = short_time_objective_intelligibility(beamformed_pred, beamformed_target, 16000).mean()
        else:
            beam_target_si_sdri = 0.0
            beam_oracle_si_sdri = 0.0
            beam_pesq = 0.0
            beam_stoi = 0.0

        if not self.supervised:
            isolated_pred = self.efficient_mixit(isolated_pred, isolated_sources, force_target=True)
            pred = self.efficient_mixit(isolated_pred, MoM)
            repeated_mix = mix[:, None, :]
            repeated_mix = repeated_mix.expand(-1, MoM.shape[1], -1)
            MoMi = (scale_invariant_signal_distortion_ratio(pred, MoM) - scale_invariant_signal_distortion_ratio(repeated_mix, MoM)).mean()

            msi = self.compute_msi(isolated_pred, isolated_sources, mix, labels)

            results = {
                "test_MoMi" : MoMi,
                "test_msi" : msi,
                "test_oracle_si_sdri" : oracle_si_sdri,
                "test_target_si_sdri" : target_si_sdri,
                "test_beam_target_si_sdri" : beam_target_si_sdri,
                "test_beam_oracle_si_sdri" : beam_oracle_si_sdri,            
            }

            if speech:
                if pesq is not None and beam_pesq is not None:
                    results.update({
                        "test_pesq" : pesq,
                        "test_stoi" : stoi,
                        "test_beam_pesq" : beam_pesq,
                        "test_beam_stoi" : beam_stoi,
                    })

            self.log_dict(results, batch_size=mix.shape[0], sync_dist=True)
        else:
            msi = self.compute_msi(isolated_pred, isolated_sources, mix, labels)

            results = {
                "test_target_si_sdri" : target_si_sdri,
                "test_oracle_si_sdri" : oracle_si_sdri,
                "test_beam_target_si_sdri" : beam_target_si_sdri,
                "test_beam_oracle_si_sdri" : beam_oracle_si_sdri,
                "test_msi" : msi,
            }

            if speech:
                if pesq is not None and beam_pesq is not None:
                    results.update({
                        "test_pesq" : pesq,
                        "test_stoi" : stoi,
                        "test_beam_pesq" : beam_pesq,
                        "test_beam_stoi" : beam_stoi,
                    })

            self.log_dict(results, batch_size=mix.shape[0], sync_dist=True)

    def log_pred(self, pred, ground_truth, mix, labels):
        """
        Debug function for logging predictions
        
        Args:
            pred (Tensor): (source, time,)
            ground_truth (Tensor): (sources, time,)
            mix (Tensor): (time,)
            mix (Tensor like): (sources, time,)

        """
        i = 0
        table = []

        for waveform_source, waveform_groundTruth, label in zip(pred, ground_truth, labels):
            table.append([
                f"{label}",
                scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean() -
                scale_invariant_signal_distortion_ratio(mix[None, ...], waveform_groundTruth[None, ...]).mean() ,
                wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
            ])
            i+=2

        self.logger.log_table(key="results", columns=self.log_columns, data=table)

    def on_test_end(self):
        """
            Log examples on test end
        """
        if isinstance(self.logger, WandbLogger):
            mix0, isolated_sources0, labels0 = self.trainer.datamodule.dataset_test_16sounds.get_serialized_sample(18, 1405)
            mix1, isolated_sources1, labels1 = self.trainer.datamodule.dataset_test_16sounds.get_serialized_sample(10, 1300)

            multimic_mix = torch.stack((mix0, mix1)).to(self.device)
            multimic_isolated_sources = torch.stack((isolated_sources0, isolated_sources1)).to(self.device)
            orig_labels = [labels0, labels1]
            labels = np.array(orig_labels).T

            # Ignore mics for now
            mix = multimic_mix[:,0]
            isolated_sources = multimic_isolated_sources[:,:,0]

            if not self.supervised:
                MoM, isolated_sources, labels = self.prepare_data_for_unsupervised(isolated_sources, labels)
            else:
                isolated_sources, labels = self.prepare_data_for_supervised(isolated_sources, labels)

            isolated_pred, pred_mask = self(mix, return_pred_mask=True)

            if multimic_mix.shape[1] > 1:
                stft_mix = self.encoder.encoder(multimic_mix)
                stft_target = self.encoder.encoder(multimic_isolated_sources[:,0])
                oracle_mask = torch.abs(stft_target[:,0]) / (torch.abs(stft_target[:,0]) + torch.abs(stft_mix[:,0]-stft_target[:,0]) + self.epsilon)
                stft_oracle_pred = (stft_mix.transpose(0,1) * oracle_mask).transpose(0,1)

                oracle_pred = self.decoder.decoder(stft_oracle_pred[:,0])
                oracle_pred = torch.nn.functional.pad(oracle_pred, (0, int(mix.shape[-1]-oracle_pred.shape[-1])), mode="constant", value=0)
                # Get Spectrograms
                stft_pred = (stft_mix.transpose(0,1) * pred_mask).transpose(0,1)
                stft_noise = stft_mix - stft_pred
                stft_oracle_noise = stft_mix - stft_oracle_pred

                # Get Spatial covariance matrices
                scm_pred = self.scm_transform(stft_pred)
                scm_pred_noise = self.scm_transform(stft_noise)
                scm_oracle = self.scm_transform(stft_oracle_pred)
                scm_oracle_noise = self.scm_transform(stft_oracle_noise)

                # Compute beamforming weight and apply it
                beamformed_pred = self.mvdr_transform(stft_mix, scm_pred, scm_pred_noise, reference_channel=0)
                beamformed_target = self.mvdr_transform(stft_target, scm_pred, scm_pred_noise, reference_channel=0)
                beamformed_oracle = self.mvdr_transform(stft_mix, scm_oracle, scm_oracle_noise, reference_channel=0)
                beamformed_oracle_target = self.mvdr_transform(stft_target, scm_oracle, scm_oracle_noise, reference_channel=0)

                # Post mask
                post_mask = pred_mask.clone()
                post_mask[post_mask<0.3] = 0.3
                beamformed_pred = beamformed_pred*pred_mask

                # Spectrogram -> Waveform
                beamformed_pred = self.decoder.decoder(beamformed_pred)
                beamformed_pred = torch.nn.functional.pad(beamformed_pred, (0, int(mix.shape[-1]-beamformed_pred.shape[-1])), mode="constant", value=0)
                beamformed_target = self.decoder.decoder(beamformed_target)
                beamformed_target = torch.nn.functional.pad(beamformed_target, (0, int(mix.shape[-1]-beamformed_target.shape[-1])), mode="constant", value=0)
                beamformed_oracle = self.decoder.decoder(beamformed_oracle)
                beamformed_oracle = torch.nn.functional.pad(beamformed_oracle, (0, int(mix.shape[-1]-beamformed_oracle.shape[-1])), mode="constant", value=0)
                beamformed_oracle_target = self.decoder.decoder(beamformed_oracle_target)
                beamformed_oracle_target = torch.nn.functional.pad(beamformed_oracle_target, (0, int(mix.shape[-1]-beamformed_oracle_target.shape[-1])), mode="constant", value=0)

                isolated_pred = self.efficient_mixit(isolated_pred, isolated_sources, force_target=True)

                label = labels[0]
                isolated_pred_logging = isolated_pred[0]
                isolated_sources = isolated_sources[0]
                beam_pred = beamformed_pred[0]
                oracle_pred = oracle_pred[0]
                beam_oracle_pred = beamformed_oracle[0]
                beam_oracle_target = beamformed_oracle_target[0]
                beamformed_target = beamformed_target[0]

                pred_energy = 10*torch.log10((isolated_pred_logging[0]**2).mean())
                beam_pred = self.rms_normalize(beam_pred, pred_energy)

                oracle_pred_energy = 10*torch.log10((oracle_pred**2).mean())
                beam_oracle_pred = self.rms_normalize(beam_oracle_pred, oracle_pred_energy)

                i = 0
                table = []

                table.append([
                        f"{label[0]}",
                        scale_invariant_signal_distortion_ratio(oracle_pred[None, ...], isolated_sources[0][None, ...]).mean() -
                            scale_invariant_signal_distortion_ratio(mix[0][None, ...], isolated_sources[0][None, ...]).mean() ,
                        wandb.Image(plot_spectrogram_from_waveform(oracle_pred[None, ...]+self.epsilon, 16000, title="Oracle prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(isolated_sources[0][None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(oracle_pred.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(isolated_sources[0].cpu().numpy(), sample_rate=16000),
                    ])
                table.append([
                        f"{label[0]}",
                        scale_invariant_signal_distortion_ratio(beam_oracle_pred[None, ...], beam_oracle_target[None, ...]).mean() -
                            scale_invariant_signal_distortion_ratio(mix[0][None, ...], isolated_sources[0][None, ...]).mean() ,
                        wandb.Image(plot_spectrogram_from_waveform(beam_oracle_pred[None, ...]+self.epsilon, 16000, title="Beamformed Oracle prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(isolated_sources[0][None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(beam_oracle_pred.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(isolated_sources[0].cpu().numpy(), sample_rate=16000),
                    ])
                table.append([
                        f"{label[0]}",
                        scale_invariant_signal_distortion_ratio(beam_pred[None, ...], beamformed_target[None, ...]).mean() -
                            scale_invariant_signal_distortion_ratio(mix[0][None, ...], isolated_sources[0][None, ...]).mean() ,
                        wandb.Image(plot_spectrogram_from_waveform(beam_pred[None, ...]+self.epsilon, 16000, title="Beamformed prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(isolated_sources[0][None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(beam_pred.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(isolated_sources[0].cpu().numpy(), sample_rate=16000),
                    ])
                for waveform_source, waveform_groundTruth in zip(isolated_pred_logging, isolated_sources):
                    table.append([
                        f"{label[i]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean() -
                            scale_invariant_signal_distortion_ratio(mix[0][None, ...], waveform_groundTruth[None, ...]).mean() ,
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=1
                self.logger.log_table(key="isolated results", columns=self.log_columns, data=table)

                wandb.log({f"Mix": wandb.Audio(mix[0].cpu().numpy(), sample_rate=16000)})

                if not self.supervised:
                    pred = self.efficient_mixit(isolated_pred, MoM)
                    preds = pred[0]
                    groundTruths = MoM[0]
                    orig_label = orig_labels[0]
                    i = 0
                    table = []

                    for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                        table.append([
                            f"{orig_label[i]}, {orig_label[i+1]}",
                            scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean() -
                            scale_invariant_signal_distortion_ratio(mix[0][None, ...], waveform_groundTruth[None, ...]).mean() ,
                            wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                            wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                            wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                            wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                        ])
                        i+=2

                    self.logger.log_table(key="results", columns=self.log_columns, data=table)

        return
    
    def validation_epoch_end(self, outputs):
        """
            Log examples on validation end every 100 epochs
        """
        outputs.clear()

        if not self.current_epoch % 100 and isinstance(self.logger, WandbLogger):
            mix0, isolated_sources0, labels0 = self.trainer.datamodule.dataset_val.get_serialized_sample(15, 1300)
            mix1, isolated_sources1, labels1 = self.trainer.datamodule.dataset_val.get_serialized_sample(10, 1300)

            mix = torch.stack((mix0, mix1)).to(self.device)
            isolated_sources = torch.stack((isolated_sources0, isolated_sources1)).to(self.device)
            orig_labels = [labels0, labels1]
            labels = np.array(orig_labels).T

            # Ignore mics for now
            mix = mix[:,0]
            isolated_sources = isolated_sources[:,:,0]

            if not self.supervised:
                MoM, isolated_sources, labels = self.prepare_data_for_unsupervised(isolated_sources, labels)
            else:
                isolated_sources, labels = self.prepare_data_for_supervised(isolated_sources, labels)

            isolated_pred = self(mix)
                   
            isolated_pred = self.efficient_mixit(isolated_pred, isolated_sources)

            label = labels[0]
            isolated_pred_logging = isolated_pred[0]
            isolated_sources = isolated_sources[0]
            i = 0
            table = []

            for waveform_source, waveform_groundTruth in zip(isolated_pred_logging, isolated_sources):
                table.append([
                    f"{label[i]}",
                    scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean() -
                        scale_invariant_signal_distortion_ratio(mix[0][None, ...], waveform_groundTruth[None, ...]).mean() ,
                    wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                    wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                ])
                i+=1
            self.logger.log_table(key="isolated results", columns=self.log_columns, data=table)

            wandb.log({f"Mix": wandb.Audio(mix[0].cpu().numpy(), sample_rate=16000)})

            if not self.supervised:
                pred = self.efficient_mixit(isolated_pred, MoM)
                preds = pred[0]
                groundTruths = MoM[0]
                orig_label = orig_labels[0]
                i = 0
                table = []

                for waveform_source, waveform_groundTruth in zip(preds, groundTruths):
                    table.append([
                        f"{orig_label[i]}, {orig_label[i+1]}",
                        scale_invariant_signal_distortion_ratio(waveform_source[None, ...], waveform_groundTruth[None, ...]).mean() -
                          scale_invariant_signal_distortion_ratio(mix[0][None, ...], waveform_groundTruth[None, ...]).mean() ,
                        wandb.Image(plot_spectrogram_from_waveform(waveform_source[None, ...]+self.epsilon, 16000, title="prediction")),
                        wandb.Image(plot_spectrogram_from_waveform(waveform_groundTruth[None, ...]+self.epsilon, 16000, title="target")),
                        wandb.Audio(waveform_source.cpu().numpy(), sample_rate=16000),
                        wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=16000),
                    ])
                    i+=2

                self.logger.log_table(key="results", columns=self.log_columns, data=table)

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

    def prepare_data_for_unsupervised(self, isolated_sources, labels=None):
        """
            Args:
                isolated_sources (Tensor): Batch, Sources, Time 
            
            Returns:
                (Tuple of tensors): (Batch, Sources/2, 2, Time), (Batch, 3, Time)
        """
        sources = isolated_sources.shape[1]
        if sources > 2:
            isolated_mix = isolated_sources.reshape(isolated_sources.shape[0], -1, 2, isolated_sources.shape[2])
            isolated_mix = torch.sum(isolated_mix, dim=2, keepdim=False)

            summed_array = torch.sum(isolated_sources[:, 2:, :], dim=1, keepdim=True)
            new_isolated_sources = torch.cat((isolated_sources[:, :2, :], summed_array), dim=1)

            new_labels = None
            if labels is not None:
                array_labels = np.array(labels).T

                new_labels: list = []
                for label in array_labels:
                    new_labels.append([str(label[0]), str(label[1]), ' '.join(map(str,label[2:]))])
        else:
            raise Exception("Unsupported amount of sources")

        return isolated_mix, new_isolated_sources, new_labels
    
    def prepare_data_for_supervised(self, isolated_sources, labels):
        """
            Args:
                isolated_sources (Tensor): Target signal needs to be the first source (Batch, Sources, Time)
            
            Returns:
                (Tensor): (Batch, 2, Time), (List): (Batch, 2)
        """

        additionnal_sources_mix = isolated_sources[:, 1:, ...]

        additionnal_sources_mix = torch.sum(additionnal_sources_mix, 1)

        new_isolated_source = torch.empty(isolated_sources.shape[0], 2, isolated_sources.shape[2], device=self.device)

        new_isolated_source[:,0] = isolated_sources[:,0]
        new_isolated_source[:,1] = additionnal_sources_mix

        array_labels = np.array(labels).T

        new_labels: list = []
        for label in array_labels:
            new_labels.append([str(label[0]),' '.join(map(str,label[1:]))])

        return new_isolated_source, new_labels
    
    @staticmethod
    def compute_msi(pred, target, mix, labels):
        """
        Compute MSI metric
        Args:
            pred (Tensor) : Predicted signal (batch, sources, time)
            target (Tensor) : Target signal (batch, sources, time)
            mix (Tensor) : Original mixture (batch, time)
            labels (list of str) : Target labels (batch, sources)
        """
        msi = scale_invariant_signal_distortion_ratio(pred, target)
        msi_count = 0
        msi_nb = 0
        for i, label in enumerate(labels):
            for j, l_batch in enumerate(label):
                if l_batch != "Nothing" and l_batch != "Nothing Nothing" and l_batch != "Nothing Nothing Nothing":
                    msi_count += msi[i, j] - scale_invariant_signal_distortion_ratio(mix[i], target[i, j])
                    msi_nb +=1
        msi = msi_count/msi_nb

        return msi
    
    def negative_SNR(self, pred, target):
        """
            Loss used in Unsupervised sound separation using mixture invariant training
            Args:
                pred (Tensor): (..., time)
                target (Tensor): (..., time)
        """
        soft_threshold = 10**(-30/10)
        loss = 10*torch.log10((target-pred)**2 + soft_threshold*(target**2) + self.epsilon) - 10*torch.log10(target**2 + self.epsilon)
        return loss.mean()
    
    @staticmethod
    def rms_normalize(x, db_value):
        """
            Normalize audio using this equation: 10*torch.log10((torch.abs(X)**2).mean()) = db_value

            Args:
                x (Tensor)
                db_value (int) : Value to normalize to in dB

            Return:
                (Tensor): normalized x
        """
        
        augmentation_gain = 10 ** (db_value/20)
        normalize_gain  = torch.sqrt(1/(torch.abs(x)**2).mean()) 
    
        return augmentation_gain * normalize_gain * x
    
    @staticmethod 
    def efficient_mixit(pred, target, force_target = False):
        """
            MixIT using least-squares
            Args:
                pred (Tensor): prediction (..., pred_sources, time)
                target (Tensor): target (..., mixtures, time)
                force_target (bool): Whether to force the first prediction into the first target mixture or not

            Return:
                (Tensor): Predictions put together to match the target mixtures (..., mixtures, time)
        """
        least_squares_result = torch.linalg.lstsq(pred.transpose(1,2), target.transpose(1,2)).solution.transpose(1,2)

        max_indices = torch.argmax(least_squares_result, dim=1)

        # Create a binary mask where 1 is set at the indices of minimum values, rest to 0
        mask = torch.zeros_like(least_squares_result)
        mask.scatter_(1, max_indices.unsqueeze(1), 1)

        if force_target:
            mask[..., 0, 0] = 1
            mask[..., 1, 0] = 0

        mixit_result = torch.bmm(mask, pred)

        return mixit_result
    
    def configure_optimizers(self):
        """
            Pytorch lightning method to provide optimizer
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, maximize=False)
        return optimizer

if __name__ == "__main__":
    conv = ConvTasNet()
    a = torch.randn(4, 4096)
    s = conv(a)
