import weakseparation
import torch
import numpy as np

sample_rate = 16000

seed = 42
if seed is not None:
    torch.manual_seed(seed) 
    np.random.seed(seed)

def main():
    dm = weakseparation.SeclumonsDataModule(
        "/home/jacob/Dev/weakseparation/library/dataset/SECL-UMONS",
        sample_rate=sample_rate,
        batch_size=16,
        num_of_workers=12
    )
    dm.prepare_data()
    dm.setup(stage="validate")

    for (mix, isolated_sources, labels) in dm.val_dataloader():
        # weakseparation.plot_spectrogram_from_spectrogram(isolated_sources[0,0], sample_rate)
        print(labels)


if __name__ == "__main__":
    main()