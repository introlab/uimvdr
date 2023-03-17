import argparse
import weakseparation
import wandb
import pytorch_lightning as pl
import torchaudio
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from asteroid.losses import MixITLossWrapper
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from tqdm import tqdm

sample_rate = 16000
supervised = False
return_spectrogram = False
seed = 42
frame_size = 512
bins = int(frame_size / 2) + 1
hop_size = int(frame_size / 2)
mics = 7
max_sources = 2
if not supervised:
    max_sources *= 2
layers = 2
hidden_dim = bins*max_sources
epochs = 1000
batch_size=8
num_of_workers=8

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080 Ti':
    batch_size=6
    num_of_workers=16
    torch.set_float32_matmul_precision('high')

def negative_si_sdr(pred, target):
    return -1*scale_invariant_signal_distortion_ratio(pred, target).mean(-1)

def main(args):

    pl.seed_everything(seed, workers=True)

    checkpoint = "/home/jacob/dev/weakseparation/library/weakseparation/src/best_models/sudo-epoch=50-val_loss=-2.37507.ckpt"
    model = weakseparation.SudoRmRf.load_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    mixit_loss = MixITLossWrapper(negative_si_sdr, generalized=False)

    stft = torchaudio.transforms.Spectrogram(
        n_fft=frame_size,
        hop_length=hop_size,
        power=None,
        window_fn=weakseparation.cuda_sqrt_hann_window
    )
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=frame_size, hop_length=hop_size, window_fn=weakseparation.cuda_sqrt_hann_window
    )
    SCM = torchaudio.transforms.PSD(normalize=False)
    mvdr_transform = torchaudio.transforms.SoudenMVDR()

    dm = weakseparation.DataModule(
        weakseparation.SeclumonsDataset,
        "/home/jacob/dev/weakseparation/library/dataset/SECL-UMONS",
        frame_size = frame_size,
        hop_size = hop_size,
        sample_rate=sample_rate,
        max_sources=max_sources,
        batch_size=batch_size,
        num_of_workers=num_of_workers,
        return_spectrogram= False
    )

    dm.setup("validate")

    with torch.no_grad():
        running_sdr = 0
        validation_dataloader = dm.val_dataloader()
        for idx, batch in enumerate(tqdm(validation_dataloader)):
            mix, isolatedSources, labels  = batch
            mix = mix.to(device)
            isolatedSources = isolatedSources.to(device)

            single_channel_isolatedSources = isolatedSources[:,:,0]
            single_channel_mix = mix[:,0][:,None,...]

            target_waveform = single_channel_isolatedSources.reshape(-1, int(max_sources/2), int(max_sources/2), isolatedSources.shape[-1])
            target_waveform = torch.sum(target_waveform, dim=2, keepdim=False)

            pred = model(single_channel_mix)

            PRED = stft(pred)
            MIX = stft(mix)
            mask = torch.abs(PRED)/(torch.abs(MIX[:,0][:,None,...]))
            mask = mask/torch.max(mask)

            MVDR = torch.zeros_like(PRED)
            for source in range(max_sources):
                scm_speech = SCM(MIX, mask[:,source])
                scm_noise = SCM(MIX, 1-mask[:,source])

                MVDR[:,source] = mvdr_transform(MIX, scm_speech, scm_noise, reference_channel=0)
            
            pred = istft(MVDR)

            loss, min_idx, parts = mixit_loss(
                pred,
                target_waveform,
                return_est=True
            )

            pred_waveform = mixit_loss.reorder_source(pred, target_waveform, min_idx, parts)

            sdr = scale_invariant_signal_distortion_ratio(pred_waveform, target_waveform).mean()

            running_sdr += sdr

        print(f"SI_SDR = {running_sdr/len(validation_dataloader)}")


    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weak multi-channel separation")
    args = parser.parse_args()

    main(args)