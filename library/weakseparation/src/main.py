import argparse
import weakseparation
import wandb
import pytorch_lightning as pl
import torchaudio
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

sample_rate = 16000
seed = 42
frame_size = 512
bins = int(frame_size / 2) + 1
hop_size = 256
mics = 7
layers = 2
hidden_dim = 128
epochs = 5

def main(args):

    pl.seed_everything(seed, workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='gru-{epoch:02d}-{val_loss:.5f}'
    )

    if args.log:
        wandb_logger = WandbLogger(project="mc-weak-separation")
    else:
        wandb_logger = None

    dm = weakseparation.SeclumonsDataModule(
        "/home/jacob/Dev/weakseparation/library/dataset/SECL-UMONS",
        frame_size = frame_size,
        hop_size = hop_size,
        sample_rate=sample_rate,
        batch_size=8,
        num_of_workers=8
    )

    model = weakseparation.GRU(bins*mics, hidden_dim, layers, mics)
    # model = weakseparation.GRU.load_from_checkpoint("/home/jacob/Dev/weakseparation/mc-weak-separation/4rxsy8rj/checkpoints/gru-epoch=00-val_loss=0.00261.ckpt")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    if args.train:
        trainer.fit(model=model, datamodule=dm)

        if args.log:
            columns = ["class", "pred spectrogram", "ground truth spectrogram", "pred audio", "ground truth audio"]

            results = trainer.predict(model=model, datamodule=dm)
            preds = results[0][0][0,:,0]
            groundTruths = results[0][1][0,:,0]
            labels = results[0][2][0]
            istft = torchaudio.transforms.InverseSpectrogram(
                    n_fft=frame_size, hop_length=hop_size, window_fn=weakseparation.sqrt_hann_window
                )
            i = 0
            table = []
            gain = 10
            for source, groundTruth, label in zip(preds, groundTruths, labels):
                waveform_source = istft(source)*gain
                waveform_groundTruth = istft(groundTruth)*gain
                class_label = weakseparation.id_to_class[label.item()]
                table.append([
                    class_label,
                    wandb.Image(weakseparation.plot_spectrogram_from_waveform(waveform_source[None, ...], sample_rate, title=class_label)),
                    wandb.Image(weakseparation.plot_spectrogram_from_waveform(waveform_groundTruth[None, ...], sample_rate, title=class_label)),
                    wandb.Audio(waveform_source.cpu().numpy(), sample_rate=sample_rate),
                    wandb.Audio(waveform_groundTruth.cpu().numpy(), sample_rate=sample_rate),
                    ])
                i+=1
            wandb_logger.log_table(key="results", columns=columns, data=table)

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

    # if args.log:
    #     wandb_logger.finalize("success")



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