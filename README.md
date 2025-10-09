# Machine Learning Approaches for road segmentation

# [Full Report Available Here](https://github.com/elsahtz2/Road-Segmentation-in-Aerial-Imagery/blob/main/Report.pdf)

<div align="center">
  <img src="https://github.com/elsahtz2/Road-Segmentation-in-Aerial-Imagery/blob/main/img/results.png" alt="Thymio Sheep" style="width:40%;">
  <br>
  <em>Results from our Models</em>
</div>

## Overview

Road segmentation from high-resolution satellite and aerial imagery is crucial for applications like transportation planning, urban development, and disaster management. This project leverages advanced deep learning models (U-Net, D-LinkNet, SegNet) for accurate pixel-wise road segmentation.

The training pipeline includes data augmentation, loss function optimization, and model comparison to address challenges such as varying road orientations, environmental conditions, and urban layouts.


## File architechture

```plaintext
📂 Project Root
│
├── README.md
├── requirements.txt
├── dataset_import.py          # Definition of custom PyTorch DataLoaders
├── fixed_rot_augmentation.py  # Dataset augmentation using fixed rotation for most common angles
                                and introduction of gaussian noise
├── mask_to_submission.py      # Converts predicted masks into submission format for AIcrowd
├── random_augmentation.py     # Dataset augmentation using random rotations for most common angles
                                and introduction of gaussian noise
├── test_pred.py               # Script for testing predictions on the provided test dataset
├── train.py                   # Training pipeline used for the diverse road segmentation models
├── report.pdf                 # Explaination of methods and results
│
└── 📂 NN                      # Neural network models
    ├── linknet_pairwise.py    # Pairwise version of LinkNet model
    ├── loss.py                # Loss functions implementations (BCE, Dice Loss, etc.)
    ├── segnet.py              # SegNet model definition
    ├── unet.py                # U-Net model definition
    └── 📂 linknet_tools       # Utilities and tools for LinkNet
```

## Data

The provided dataset can be download from AIcrowd using this link : https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files

- **Training**: 100 satellite images (400×400 RGB) with corresponding ground-truth masks.
- **Testing**: 50 satellite images (608×608 RGB).  
  The dataset is loaded using functions from dataset_import.py

### Aditional dataset

[**Dataset from Lucci et al. in Learning Aerial Image Segmentation from Online Maps**](https://ieeexplore.ieee.org/document/7987710): Features diverse scenes from Chicago, Zurich, Berlin, Paris, and Tokyo, with a mix of urban and rural environments. Its diversity aligns closely with the characteristics of our test set.

### Data augmentation and preprocessing

To improve model generalization, we applied the following augmentations:

1. **Fixed Rotations and Gaussian Noise**:

   - Implemented in `fixed_rot_augmentation.py`.
   - Applies common angles (e.g., 15°, 30°, 45°, 90°) and introduces Gaussian noise.

2. **Random Rotations and Gaussian Noise**:

   - Implemented in `random_augmentation.py`.
   - Applies random rotation angles and Gaussian noise for better variability.

3. **Additional Techniques**:
   Both `fixed_rot_augmentation.py` and `random_augmentation.py` also implement:
   - Horizontal and vertical flipping.
   - Brightness and contrast adjustments.

These augmentations help address biases in the original dataset and improve the model's ability to generalize across diverse road orientations and lighting conditions.

# Usage

## Install requirements

In order to use this project, start by cloning the repository. Make sure you have python 3.9 or above and install all dependancies `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

## Data organisation and preprocessing

First download the original and the additional dataset and organize it such as:

```plaintext
📂 dataset
│
└── 📂 training
    ├── 📂 images                <- Original training images
    └── 📂 groundtruth           <- Ground truth masks
```

Then running `fixed_rot_augmentation.py` and `random_augmentation.py` will create the augmented dataset such as:

```plaintext
📂 dataset
│
├── 📂 training
│   ├── 📂 images               <- Original training images
│   └── 📂 groundtruth          <- Ground truth masks
│
└── 📂 augmented_dataset
    ├── 📂 images               <- Output augmented images
    └── 📂 groundtruth          <- Output augmented masks
```

You should only modify `dataset_dir` in both script to match the path of your dataset folder.

```bash
python fixed_rot_augmentation.py
python random_augmentation.py
```

# Training and Running the Models

This guide explains how to train the models and create a submission for AIcrowd using the provided scripts.

---

## Training the Models

To train the models, the following parameters can be edited in **`train.py`**:

### Editable Parameters
- **`model_name`**: Choose the model architecture from the following options:
  - `"UNet"`
  - `"LinkNet34"`
  - `"SegNet"`
  - `"NL_LinkNet_EGaussian"`
- **`batch_size`**: Set the desired batch size for training.
- **`epochs`**: Specify the maximum number of epochs for training.
- **`learning_rate`**: Define the initial learning rate for the optimizer.
- **`loss_name`**: Choose the loss function from the following options:
  - `'bce'`, `'weighted_bce'`, `'balanced_bce'`, `'focal'`
  - `'jaccard'`, `'dice'`, `'squared_dice'`, `'log_cosh_dice'`
  - `'tversky'`, `'focal_tversky'`, `'bce_dice'`, `'combo'`

### File Locations
- **`save_path`**: Modify **line 46** in `train.py` to change the directory where trained model weights are saved.
- **`plots_dir`**: Modify **line 256** in `train.py` to set the directory for saving training plots (e.g., loss and accuracy curves).




# Running the Pipeline

The `run.py` script combines the training, prediction, and submission generation steps.

---

## Options for Running `run.py`

### Train the Model and Generate Predictions
If no pre-trained model is provided, the script will:
1. Train the model using the parameters specified in `train.py`.
2. Use the trained model to predict masks for the test set.
3. Generate a submission file for AIcrowd.

#### Example Command
```bash
python run.py --model_name "UNet"
```

## Skip Training and Use a Pre-Trained Model

If a pre-trained model file (`.pt`) is provided, the script will:
1. Skip the training phase.
2. Use the provided model to predict masks for the test set.
3. Generate a submission file.

### Example Command
```bash
python run.py --model_path models/trained_model_UNet.pt
```

## Additional Arguments for run.py
- **`--test_set`**: Specify the path to the test dataset (default: dataset/test_set_images).
- **`--output_masks`**: Directory to save the predicted test masks (default: test_set_masks).
- **`--submission_output`**: File path for the generated submission (default: submission.csv).