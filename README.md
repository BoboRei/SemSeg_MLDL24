# Real-Time Semantic Segmentation via FDA and Adversarial Domain Adaptation

This repository contains code for semantic segmentation using DeepLabv2 and BiSeNet, with implementations for Unsupervised Domain Adaptation (UDA) techniques such as Adversarial Learning and Fourier Domain Adaptation (FDA).

## Repository Structure

### Dataset
The `dataset` folder contains scripts for handling the GTA5 and Cityscapes datasets.
- `GTA5.py`: Script for the GTA5 dataset class.
- `Cityscapes.py`: Script for the Cityscapes dataset class.

### Models
The `models` folder includes the model architectures for BiSeNet and DeepLabv2.
- `BiSeNet.py`: Class definition for the BiSeNet model.
- `DeepLabv2.py`: Class definition for the DeepLabv2 model.

### Train DeepLab
The `train_deeplab` folder contains the script for training and validating DeepLabv2 on the Cityscapes dataset.
- `train_deeplab.py`: Script to train and validate DeepLabv2 on Cityscapes.

### Train BiSeNet
The `train_bisenet` folder contains various scripts for training and validating BiSeNet under different conditions.
- `bisenet_cityscapes.py`: Train and validate BiSeNet on Cityscapes.
- `bisenet_gta_to_cityscapes.py`: Train BiSeNet on GTA5 and validate on Cityscapes.
- `adv_bisenet_gta_to_cityscapes.py`: Train BiSeNet on GTA5 and validate on Cityscapes using adversarial learning for UDA.
- `adv_fda_bisenet_gta_to_cityscapes.py`: Train BiSeNet on GTA5 and validate on Cityscapes using a combination of adversarial learning and FDA for UDA.
- `ensemble_adv_bisenet_gta_to_cityscapes.py`: Validate BiSeNet on Cityscapes using ensemble predictions from three models trained with different `beta` values (hyperparameter of FDA). This stabilizes the predictions.

### Other Scripts
- `Discriminator.py`: Contains the class for the discriminator network used in the context of adversarial learning.
- `fda.py`: Includes functions for applying Fourier Domain Adaptation.
- `utils.py`: Contains utility functions such as mIOU calculation and learning rate scheduler.