import argparse
import weakseparation
import wandb
import pytorch_lightning as pl
import torchaudio
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

target_class = "Bark"
non_mixing_classes = ["Dog"]
# non_mixing_classes = [""]
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
beta = 0.25
gamma = 1
classification_percentage = 0

if torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080 Ti':
    batch_size=16
    num_of_workers=16
    torch.set_float32_matmul_precision('high')

def main(args):

    pl.seed_everything(seed, workers=True)

    # trainer.checkpoint_callback.best_model_path
    checkpoint_callback = ModelCheckpoint(
        monitor='val_target_SI_SDR',
        mode = 'max',
        filename='convtasnet-{epoch:02d}-{val_loss:.5f}-{val_target_SI_SDR:.3f}'
    )

    if args.log:
        wandb_logger = WandbLogger(project="mc-weak-separation")
    else:
        wandb_logger = CSVLogger("/home/jacob/dev/weakseparation/logs")

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
        weakseparation.customDataset.CustomDataset,
        "/home/jacob/dev/weakseparation/library/dataset/Custom/separated",
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

    model = weakseparation.ConvTasNet(
        N=frame_size, 
        H=bins, 
        activate="softmax", 
        num_spks=num_speakers, 
        supervised=supervised,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        classi_percent=classification_percentage,
        learning_rate=learning_rate
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        deterministic=True,
        log_every_n_steps=5,
        gradient_clip_algorithm="value",
        gradient_clip_val=5
    )

    if args.train:
        trainer.fit(model=model, datamodule=dm)

    if args.predict:
        if args.predict[-5:] == ".ckpt":
            print("Starting Testing")
            model = weakseparation.ConvTasNet.load_from_checkpoint(
                checkpoint_path=args.predict
            )
            trainer.test(model=model, datamodule=dm)

            print("Ending Testing")
        else:
            print("Starting Testing")
            trainer.test(model=model, datamodule=dm, ckpt_path="best")
            print("Ending Testing")

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
        help="If true, will test the model",
        nargs="?",
        default=False,
        const="best",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="If true, will log to weights and biases",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)