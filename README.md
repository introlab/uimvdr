# UIMVDR

This is the repository for UIMVDR.

## Abstract

Recently, neural networks have gained popularity for sound enhancement and separation. Despite their great performances, these results usually rely on large labelled datasets. Data available online mostly consists of single-channel, non-isolated recordings, making them unsuitable for supervised training and processing multi-channel audio. Multi-channel approaches have outperform single channel approaches as they can exploit both spatial and spectral features. Unsupervised Improved Minimum Variation Distortionless Response (UIMVDR) enables multi-channel sound enhancement and separation to leverage in-the-wild single-channel data through unsupervised training and beamforming. Results suggests that, UIMVDR is more robust to new domains and improves separation performances compared to supervised networks, particularly in scenarios where supervised data is limited. By using data available online, it also reduces the effort required in gathering data for multi-channel approaches.

## Get the code

```bash
# Clone with git in a terminal
git clone https://github.com/introlab/uimvdr.git
# Go in the root folder
cd uimvdr
# Install the dependencies
pip install -r requirements.txt
```

## Get the Multi-Channel Free Sound Test Dataset (MCFSTD)

[Zenodo link](https://zenodo.org/records/10601318) 

## Pretrained models

Get pretrained models on [Google Drive](https://drive.google.com/drive/folders/1ERosQmD0yiLmH5JYttGRuyTanNQvH3XI?usp=sharing)

## Improvements

Send us your comments/suggestions to improve the project in ["Issues"](https://github.com/introlab/weakseparation/issues).

## Authors

* Jacob Kealey (@JacobKealey)
* François Grondin (@FrancoisGrondin)

## Licence

* [BSD-3](LICENSE)

## Acknowledgments

Thanks to Jusper Lee for his pytorch implementation of the ConvTasNet: https://github.com/JusperLee/Conv-TasNet

The work done here was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and by the Fonds de Recherche du Québec en Nature et Technologies (FRQNT).

![IntRoLab](docs/IntRoLab.png)

[IntRoLab - Laboratoire de robotique intelligente / interactive / intégrée / interdisciplinaire @ Université de Sherbrooke](https://introlab.3it.usherbrooke.ca)
