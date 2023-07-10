# SECL-UMONS-Dataset
Prensentation of the SECL-UMONS dataset for sound event classification and localization.

## Authors

- Mathilde Brousmiche (<mathilde.brousmiche@umons.ac.be>)
- Stéphane Dupont (<stephane.dupont@umons.ac.be>)
- Jean Rouat (<jean.rouat@usherbrooke.ca>)

## Download link

The dataset is available on [zenodo](https://zenodo.org/record/3632377#.Xr5HjhbgrCI)

## Description

SECL_UMONS is a dataset created for the sound event classification and localization tasks in the context of office environments. The dataset is composed of multichannel recordings of isolated acoustic events typically found in an office environment.

The dataset is divided into two parts regardless of the number of events in the sequence: unilabel sequences (one event per sequence) or multilabel sequences (two simultaneous events per sequence). The unilabel sequences comprise 2662 sequences of 3 seconds composed of only one event. The multilabel sequences 2724 sequences of 4 seconds composed of two events. The recordings are sampled at 44100 Hz.

The dataset comprises 11 classes:

* chair movement
* cup drop off
* Furniture's drawer
* hand clap
* keyboard
* knock
* phone ring
* radio
* speaker
* step
* whistle

Each class comprises different subclasses. The difference between subclasses is either the use of a different object belonging to the same class or a different participant performing the action. 

The sound events are recorded in two different rooms :
* Room 1 : Metting room with chairs, tables and windows
<p align="center">
   <img src="images/room1_picture.jpg" width="400" title="Room 1">
</p>
<p align="center">
<img src="images/room1_schema.png" width="600" title="Room 1 schema">
</p>

* Room 2 : Experimental room with chairs, tables and high ceiling.
<p align="center">
   <img src="images/room2_picture.jpg" width="400" title="Room 2">
</p>
<p align="center">
<img src="images/room2_schema.png" width="600" title="Room 2 schema">
</p>

Each event class occurs at different realistic positions in the room. 
* chair movement : Red
* cup drop off : Orange
* Furniture's drawer : Purple
* hand clap : Red + Green
* keyboard : Orange
* knock : Purple
* phone ring : Orange
* radio : Purple
* speaker : Red + Green
* step : Purple + Green
* whistle : Purple + Green


All the sound events were collected in Belgium by Mons University between 12/2018 - 02/2019. The acoustic events are recorded with a [UMA-8-SP microphone array](https://www.minidsp.com/products/usb-audio-interface/uma-8-sp-detail) and Audacity software.


## Recording Procedure

Several minute long recordings (named session) of several events are recorded. Afterward, the sequences of interest containing only the event sound are extracted. One session is realized for each subclass. During one session, the participant makes the event at each possible position marked beforehand. A script is run for each session. The script determines when and where the events have to occur. It determines also when to move between two events to avoid the presence of the noise of the participant movement in the extracted sequences. The script progression is shown on a screen in the room. Afterward, sequences of interest are automatically extracted from the session recordings thanks to the time determined by the script.

## Naming Convention

The unilabel sequence recordings in the dataset follow the naming convention:

    room[room number]_array_micro[micro number]_[class number]_[subclass number]_[recording number per subclass].wav

The multilabel sequence recordings in the dataset follow the naming convention:

    room[room number]_array_micro[micro number]_[first class number]_[second class number]_[recording number per duo of classes].wav


## Reference labels

As labels for the unilabel sequences, for each subclass in the dataset, we provide a csv format file with the naming convention:

    room[room number]_metadata_[class number]_[subclass number].csv

These files enlists the file name, the sound event class, the number of the sound event class, the subclass, the coordinates x,y,z in the room, the number of the room.

In the case of the label "spearker", a additional information is added to the metadata : the speaker text


As labels for the multilabel sequences, for each duo of events in the dataset, we provide a csv format file with the naming convention:

    room[room number]_metadata_[first class number]_[second class number].csv

These files enlists the file name, the first sound event class, the number of the first sound event class, the first subclass, the number of the first subclass, the coordinates x,y,z of the first class in the room, the second sound event class, the number of the second sound event class, the second subclass, the number of the second subclass, the coordinates x,y,z of the second class in the room, the number of the room.

In the case of the label "spearker", a additional information is added to the metadata : the speaker text


## Task setup

Two different splits of the data into training et test sets are proposed for the unilabel sequences :

* split1 : 44 sequences are chosen randomly in each class (22 in each room) to constitute the test set. The rest of the sequences constitutes the training set. Therefore, we obtain a total of 484 sequences for the test set and 2178 sequences for the training test.
* split2 (test the generalization ability): one subclass of each class is used as test data. Therefore, we obtain a total of 671 sequences for the test set and 1991 sequences for the training test.

One split of the data into training et test sets is proposed for the multilabel sequences :

* split : 10 sequences are chosen randomly for each possible duo between classes (5 in each room) to constitute the test set. The rest of the sequences constitutes
the training set. Therefore, we obtain a total of 545 sequences for the test set and 2179 sequences for the training test.

## References

To cite this dataset:
```
@inproceedings{brousmiche2020secl,
  title={{SECL-UMons Database for Sound Event Classification and Localization}},
  author={Brousmiche, Mathilde and Rouat, Jean and Dupont, St{\'e}phane},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={756--760},
  year={2020},
  organization={IEEE}
}
```

## Acknowledgments
We would like to thank our colleagues from the numediart institute who participated in the creation of the dataset.

Thanks to CHISTERA IGLU and the European Regional Development Fund (ERDF) for funding