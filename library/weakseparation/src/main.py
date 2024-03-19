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
        if args.resume_training:
            keyword = "last"
        else:
            keyword = "convtasnet"
        ckpt_path = [s for s in files if keyword in s][0]
        run_id = ckpt_path.split("/")[-3]
        root = args.ckpt_path.split("/")[0:-2]
        root = "/".join(root)
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(root, "wandb")):
            for dir in dirnames:
                dir_id = dir.split("-")[-1]
                if run_id == dir_id:
                    run_path = os.path.join(dirpath,dir)

        config_path = os.path.join(run_path, "files/config.yaml")

    speech_set = {"Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking"}

    if "Bark" in args.target_class:
        args.branch_class = "Domestic animals, pets"
        args.non_mixing_classes = ["Dog"]
    elif not set(args.target_class).isdisjoint(speech_set) or "Speech" in args.target_class:
        args.target_class = ["Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking"]
        args.branch_class = "Human voice"
        args.non_mixing_classes = ["Human voice"]
    elif "Piano" in args.target_class:
        args.branch_class = "Musical instrument"
        args.non_mixing_classes = ["Keyboard (musical)"]
    else:
        print("Please make sure to define the branch class and non_mixing_classes argument as it is not a usual target class")

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
        
    resume_training = args.resume_training
    target_class = logger.experiment.config["target_class"]
    non_mixing_classes = logger.experiment.config["non_mixing_classes"]
    branch_class = logger.experiment.config["branch_class"]
    sample_rate = logger.experiment.config["sample_rate"]
    supervised = logger.experiment.config["supervised"]
    nb_of_seconds = logger.experiment.config["secs"]
    epochs = args.epochs
    learning_rate = logger.experiment.config["learning_rate"]
    batch_size = logger.experiment.config["batch_size"]
    num_of_workers = logger.experiment.config["num_of_workers"]
    beta = logger.experiment.config["beta"]
    gamma = logger.experiment.config["gamma"]
    kappa = logger.experiment.config["kappa"]
    try:
        audioset = logger.experiment.config["audioset"]
    except:
        audioset = False
    
    # logger = CSVLogger("/home/jacob/dev/weakseparation/logs")
    # batch_size = 1
    isolated = True if supervised else False
    return_spectrogram = False
    seed = 17
    frame_size = 1024
    hop_size = int(frame_size / 2)
    max_sources = 4
    num_speakers = 2 if supervised else 3 #pred by NN

    pl.seed_everything(seed, workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_target_SI_SDRi',
        mode = 'max',
        filename='convtasnet-{epoch:02d}-{val_loss:.5f}-{val_target_SI_SDRi:.3f}',
        save_last=True
    )

    if audioset:
        dataset_class = weakseparation.AudioSetDataset
        dataset_path = os.path.join(args.dataset_path, "/media/jacob/2fafdbfa-bd75-431c-abca-c664f105eef9/audioset")
    else:
        dataset_class = weakseparation.FSD50KDataset
        dataset_path= args.dataset_path

    custom_dataset_path = os.path.join(args.dataset_path, "Custom", "separated")
    dm = weakseparation.DataModule(
        dataset_class,
        dataset_path,
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

    model = weakseparation.TDCNPP(
        N=frame_size, 
        activate="softmax", 
        num_spks=num_speakers, 
        supervised=supervised,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        learning_rate=learning_rate
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=-1,
        strategy = DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
        logger=logger,
        deterministic=True,
        log_every_n_steps=5,
        resume_from_checkpoint=ckpt_path if resume_training else None
    )

    if args.train:
        trainer.fit(model=model, 
                    datamodule=dm,
                    ckpt_path=ckpt_path if resume_training else None)

    if args.predict:
        if not args.resume_training and os.path.exists(ckpt_path):
            print(f"Starting Testing for {ckpt_path}")
            model = weakseparation.TDCNPP.load_from_checkpoint(
                checkpoint_path=ckpt_path
            )
            trainer.test(model=model, datamodule=dm)

            print(f"Ending Testing for {ckpt_path}")
        else:
            print("Starting Testing")
            trainer.test(model=model, datamodule=dm, ckpt_path="best")
            print("Ending Testing")

    if args.example:
        dm.setup("test")
        paths = [
            "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/E/Speech/24.wav",
            "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/G/Bark/5.wav",
            "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/H/Church_bell/10.wav",
            "/home/jacob/dev/weakseparation/library/dataset/Custom/separated/1002/16sounds/B/Thunder/7.wav",
        ]
        mix, isolated_sources, labels = dm.dataset_test_16sounds.get_personalized_sample(paths)
        if not args.resume_training and os.path.exists(ckpt_path):
            print(f"Logging example for {ckpt_path}")
            model = weakseparation.TDCNPP.load_from_checkpoint(
                checkpoint_path=ckpt_path,
            )
            model.to('cuda' if torch.cuda.is_available() else 'cpu')
            multimic_mix = torch.stack((mix, mix))
            multimic_isolated_sources = torch.stack((isolated_sources, isolated_sources))
            orig_labels = [labels, labels]
            model.log_example(multimic_mix, multimic_isolated_sources, orig_labels, logger)
        else:
            print("Logging example")
            multimic_mix = torch.stack((mix, mix)).to(model.device)
            multimic_isolated_sources = torch.stack((isolated_sources, isolated_sources)).to(model.device)
            orig_labels = [labels, labels]
            model.log_example(multimic_mix, multimic_isolated_sources, orig_labels)

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
    parser.add_argument(
        "--example",
        help="If true, will log to an example to weight and biases",
        action="store_true",
    )
    parser.add_argument(
        "--audioset",
        help="If true, will use Audioset else, will use FSD50K",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)