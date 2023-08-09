import argparse
import weakseparation
import wandb
import pytorch_lightning as pl
import torchaudio
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

target_class = "Bark"
non_mixing_classes = ["Dog"]
sample_rate = 16000
supervised = False
return_spectrogram = False
seed = 42
learning_rate = 1e-4
frame_size = 1024
bins = int(frame_size / 2) + 1
hop_size = int(frame_size / 2)
mics = 5
max_sources = 2
num_speakers = 2
if not supervised:
    max_sources *= 2
    num_speakers = max_sources
layers = 2
hidden_dim = bins*max_sources
epochs = 4005
batch_size=16
num_of_workers=8
alpha = 1
beta = 1
delta = 0.5
classification_percentage = 1

if torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080 Ti':
    batch_size=8
    num_of_workers=16
    torch.set_float32_matmul_precision('high')

def main(args):

    pl.seed_everything(seed, workers=True)

    # trainer.checkpoint_callback.best_model_path
    checkpoint_callback = ModelCheckpoint(
        monitor='val_target_SI-SDR',
        mode = 'max',
        filename='convtasnet-{epoch:02d}-{val_loss:.5f}-{target_SI-SDR:.3f}'
    )

    if args.log:
        wandb_logger = WandbLogger(project="mc-weak-separation")
    else:
        wandb_logger = None

    # dm = weakseparation.DataModule(
    #     weakseparation.FUSSDataset,
    #     "/home/jacob/dev/weakseparation/library/dataset/FUSS/FUSS_ssdata/ssdata",
    #     frame_size = frame_size,
    #     hop_size = hop_size,
    #     sample_rate=sample_rate,
    #     max_sources=max_sources,
    #     batch_size=batch_size,
    #     num_of_workers=num_of_workers,
    #     return_spectrogram=return_spectrogram,
    # )

    dm = weakseparation.DataModule(
        weakseparation.FSD50K.FSD50KDataset,
        "/home/jacob/dev/weakseparation/library/dataset/FSD50K",
        target_class = target_class,
        non_mixing_classes = non_mixing_classes,
        frame_size = frame_size,
        hop_size = hop_size,
        sample_rate=sample_rate,
        max_sources=max_sources,
        batch_size=batch_size,
        num_of_workers=num_of_workers,
        return_spectrogram=return_spectrogram,
        supervised=supervised,
    )

    # dm = weakseparation.DataModule(
    #     weakseparation.LibrispeechDataset,
    #     "/home/jacob/dev/weakseparation/library/dataset/Librispeech",
    #     frame_size = frame_size,
    #     hop_size = hop_size,
    #     sample_rate=sample_rate,
    #     max_sources=max_sources,
    #     batch_size=batch_size,
    #     num_of_workers=num_of_workers,
    #     return_spectrogram=return_spectrogram,
    # )

    # model = weakseparation.GRU(bins*mics, hidden_dim, layers, mics, max_sources)
    # model = weakseparation.ASTTransformer(1, max_sources, mics, supervised=supervised) 
    # model = weakseparation.SudoRmRf(1, max_sources, mics, supervised=supervised)
    # model = weakseparation.UNetMixIT(2, max_sources, mics, supervised=supervised)
    # model = weakseparation.RNN.GRU(input_size=514, hidden_size=256, num_layers=4, mics=1, sources=max_sources)
    # model = weakseparation.GRU.load_from_checkpoint("/home/jacob/Dev/weakseparation/mc-weak-separation/4rxsy8rj/checkpoints/gru-epoch=00-val_loss=0.00261.ckpt")
    model = weakseparation.ConvTasNet(
        N=frame_size, 
        H=bins, 
        activate="softmax", 
        num_spks=num_speakers, 
        supervised=supervised,
        alpha=alpha,
        beta=beta,
        delta=delta,
        classi_percent=classification_percentage,
        learning_rate=learning_rate
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=1
    )

    if args.train:
        trainer.fit(model=model, datamodule=dm)

    if args.predict:
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

    if args.log:
        wandb.join()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weak multi-channel separation")
    parser.add_argument(
        "-t",
        "--train",
        help="If true, will train the model",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--predict",
        help="If true, will make a prediction and save the result",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="If true, will log to weights and biases",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)