import weakseparation
import torch
import numpy as np
import pytorch_lightning as pl
import torchaudio

sample_rate = 16000
seed = 42
frame_size = 512
bins = int(frame_size / 2) + 1
hop_size = 256
mics = 7
layers = 2
hidden_dim = 128
epochs = 5

pl.seed_everything(seed, workers=True)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

if seed is not None:
    torch.manual_seed(seed) 
    np.random.seed(seed)

def main():
    dm = weakseparation.SeclumonsDataModule(
        "/home/jacob/Dev/weakseparation/library/dataset/SECL-UMONS",
        frame_size = frame_size,
        hop_size = hop_size,
        sample_rate=sample_rate,
        batch_size=8,
        num_of_workers=8
    )
    model = weakseparation.GRU(bins*mics, hidden_dim, layers, mics)
    # dm.prepare_data()
    # dm.setup(stage="validate")

    # for (mix, isolated_sources, labels) in dm.val_dataloader():
    #     # weakseparation.plot_spectrogram_from_spectrogram(isolated_sources[0,0], sample_rate)
    #     print(labels)
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1)
    trainer.fit(model=model, datamodule=dm)

    predictions = trainer.predict(model=model, datamodule=dm)
    predictions = predictions[0][0]
    istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=frame_size, hop_length=hop_size, window_fn=weakseparation.sqrt_hann_window
        )
    i = 0
    for source in predictions:
        waveform = istft(source)
        torchaudio.save(f'./output{i}.wav', waveform, sample_rate)
        i+=1



if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()