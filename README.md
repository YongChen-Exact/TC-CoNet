# TC-CoNet
This repository contains the supported pytorch code and configuration files to reproduce of TC-CoNet.

<<<<<<< HEAD
![TC-CoNet Architecture](img/Architecture_overview.jpg?raw=true)
=======
![VT-UNet Architecture](img/Architecture_overview.jpg?raw=true)
>>>>>>> b34c713eb2d819ba9319f2710977b451185b3813



Parts of codes are borrowed from [nn-UNet](https://github.com/MIC-DKFZ/nnUNet). For detailed configuration of the dataset, please refer to [nn-UNet](https://github.com/MIC-DKFZ/nnUNet).

## Environment

Please prepare an environment with Python 3.7, Pytorch 1.7.1, and Windows 10.

## Dataset Preparation

Datasets can be acquired via following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**Dataset III**
[Brain_tumor](http://medicaldecathlon.com/)

## Preprocess Data

- TCCoNet_convert_decathlon_task -i D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_raw\TCCoNet_raw_data
- TCCoNet_plan_and_preprocess -t 2

## Functions of scripts

- **Network architecture:**
  - TCCoNet\TCCoNet\network_architecture\TCCoNet_acdc.py``
  - TCCoNet\TCCoNet\network_architecture\TCCoNet_synapse.py``
  - TCCoNet\TCCoNet\network_architecture\TCCoNet_tumor.py``
  - TCCoNet\TCCoNet\network_architecture\TCCoNet_heart.py``
  - TCCoNet\TCCoNet\network_architecture\TCCoNet_lung.py``
- **Trainer for dataset:**
  - TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_acdc.py``
  - TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_synapse.py``
  - TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_tumor.py``
  - TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_heart.py``
  - TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_lung.py``

## Train Model

- python run_training.py  3d_fullres  TCCoNetTrainerV2_TCCoNet_synapse 2 0


## Test Model

- python predict.py -i D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_raw\TCCoNet_raw_data\Task002_Synapse\imagesTs
  -o D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_raw\TCCoNet_raw_data\Task002_Synapse\imagesTs_infer
  -m D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_trained_models\TCCoNet\3d_fullres\Task002_Synapse\TCCoNetTrainerV2_TCCoNet_synapse__TCCoNetPlansv2.1
  -f 0

- python TCCoNet/inference_synapse.py

## Acknowledgements

This repository makes liberal use of code from [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

