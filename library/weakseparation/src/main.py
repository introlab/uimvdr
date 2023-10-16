import os
import argparse
import weakseparation
import wandb
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, CSVLogger

if torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080 Ti':
    torch.set_float32_matmul_precision('high')

def main(args):
    ckpt_path=""
    if args.ckpt_path is not None:
        files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(args.ckpt_path) for f in filenames]
        ckpt_path = files[0]
        run_id = ckpt_path.split("/")[-3]
        root = args.ckpt_path.split("/")[0:-2]
        root = "/".join(root)
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(root, "wandb")):
            for dir in dirnames:
                dir_id = dir.split("-")[-1]
                if run_id == dir_id:
                    run_path = os.path.join(dirpath,dir)

        config_path = os.path.join(run_path, "files/config.yaml")

    if args.log:
        if args.ckpt_path is not None:
            logger = WandbLogger(project="mc-weak-separation", save_dir=args.log_path, config=config_path)
        else:   
            logger = WandbLogger(project="mc-weak-separation", save_dir=args.log_path, config=args)
    else:
        if args.ckpt_path is not None:
            logger = WandbLogger(project="mc-weak-separation", save_dir=args.log_path, offline=True, config=config_path)
        else:   
            logger = WandbLogger(project="mc-weak-separation", save_dir=args.log_path, offline=True, config=args)
        
    resume_training = logger.experiment.config["resume_training"]
    target_class = logger.experiment.config["target_class"]
    non_mixing_classes = logger.experiment.config["non_mixing_classes"]
    branch_class = logger.experiment.config["branch_class"]
    sample_rate = logger.experiment.config["sample_rate"]
    supervised = logger.experiment.config["supervised"]
    nb_of_seconds = logger.experiment.config["secs"]
    epochs = logger.experiment.config["epochs"]
    learning_rate = logger.experiment.config["learning_rate"]
    batch_size = logger.experiment.config["batch_size"]
    num_of_workers = logger.experiment.config["num_of_workers"]
    alpha = logger.experiment.config["alpha"]
    beta = logger.experiment.config["beta"]
    gamma = logger.experiment.config["gamma"]
    kappa = logger.experiment.config["kappa"]
    classification_percentage = logger.experiment.config["classification_percentage"]

    # logger = CSVLogger("/home/jacob/dev/weakseparation/logs")
    # batch_size = 1
    isolated = True if supervised else False
    return_spectrogram = False
    seed = 17
    frame_size = 1024
    bins = int(frame_size / 2) + 1
    hop_size = int(frame_size / 2)
    max_sources = 4
    num_speakers = 2 if supervised else 3 #pred by NN

    pl.seed_everything(seed, workers=True)

    if not supervised:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_MoMi',
            mode = 'max',
            filename='convtasnet-{epoch:02d}-{val_loss:.5f}-{val_MoMi:.3f}'
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_target_SI_SDRi',
            mode = 'max',
            filename='convtasnet-{epoch:02d}-{val_loss:.5f}-{val_target_SI_SDRi:.3f}'
        )

    fsd50k_path = os.path.join(args.dataset_path, "FSD50K")
    custom_dataset_path = os.path.join(args.dataset_path, "Custom", "separated")
    dm = weakseparation.DataModule(
        weakseparation.FSD50K.FSD50KDataset,
        fsd50k_path,
        weakseparation.customDataset.CustomDataset,
        custom_dataset_path,
        target_class=target_class,
        non_mixing_classes=non_mixing_classes,
        branch_class=branch_class,
        frame_size=frame_size,
        hop_size=hop_size,
        sample_rate=sample_rate,
        max_sources=max_sources,
        nb_of_seconds=nb_of_seconds,
        batch_size=batch_size,
        num_of_workers=num_of_workers,
        return_spectrogram=return_spectrogram,
        supervised=supervised,
        isolated=isolated
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
        kappa=kappa,
        classi_percent=classification_percentage,
        learning_rate=learning_rate
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES")),
        progress_bar_refresh_rate=0,
        strategy = DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        logger=logger,
        deterministic=False if classification_percentage else True,
        log_every_n_steps=5,
        resume_from_checkpoint=ckpt_path if resume_training else None
    )

    if args.train:
        trainer.fit(model=model, datamodule=dm)

    if args.predict:
        if not args.resume_training and os.path.exists(ckpt_path):
            print(f"Starting Testing for {ckpt_path}")
            model = weakseparation.ConvTasNet.load_from_checkpoint(
                checkpoint_path=ckpt_path
            )
            trainer.test(model=model, datamodule=dm)

            print(f"Ending Testing for {ckpt_path}")
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
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--ckpt_path",
        help="If true, will test the model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resume_training",
        help="If true, will test the model, need to pass ckpt_path with this flag",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="If true, will log to weights and biases",
        action="store_true",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        nargs='*',
        default=["Bark"],
        help="Target class for training"
    )
    parser.add_argument(
        '--non_mixing_classes',
        nargs='*',
        type=str,
        default=["Dog"],
        help="Classes that should not be in additionnal sources in the dataset",
    )
    parser.add_argument(
        '--branch_class',
        type=str,
        default="Domestic animals, pets",
        help="There shall not be classes outside of this branch for the target in the dataset. Can be root or child of root ex: Animal our Wild Animals",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate"
    )
    parser.add_argument(
        "-s",
        "--supervised",
        help="If true, will train in a supervised manner",
        action="store_true",
    )
    parser.add_argument(
        "--secs",
        type=int,
        default=5,
        help="Number of seconds in each audio sample",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4004,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_of_workers",
        type=int,
        default=16,
        help="Number of threads to load data"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Classification weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="The power of the energy ",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="The weigth of the energy",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="The weight of sparcity",
    )
    parser.add_argument(
        "--classification_percentage",
        type=float,
        default=0.0,
        help="Percentage of classification used in training",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/home/jacob/dev/weakseparation/logs",
        help="Logging path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/jacob/dev/weakseparation/library/dataset",
        help="Logging path",
    )



    args = parser.parse_args()
    main(args)