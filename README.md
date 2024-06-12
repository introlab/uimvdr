# UIMVDR

This is the repository for UIMVDR.

## Abstract

Neural networks have recently become the dominant approach to sound separation. Their good performance relies on large datasets of isolated recordings. For speech and music, isolated single channel data are readily available; however, the same does not hold in the multi-channel case, and with most other sound classes. Multi-channel methods have the potential to outperform single channel approaches as they can exploit both spatial and spectral features, but the lack of training data remains a challenge. We propose unsupervised improved minimum variation distortionless response (UIMVDR), which enables multi-channel separation to leverage in-the-wild single-channel data through unsupervised training and beamforming. Results show that UIMVDR generalizes well and improves separation performance compared to supervised models, particularly in cases with limited supervised data. By using data available online, it also reduces the effort required to gather data for multi-channel approaches.

## Get the code

```bash
# Clone with git in a terminal
git clone https://github.com/introlab/uimvdr.git
# Go in the root folder
cd uimvdr
# Install the dependencies
pip install -r requirements.txt
```

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
