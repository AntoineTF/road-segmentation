import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ---------------------------------------------------------------
# AugmentedDataset: Training/Validation Dataset with Image Masks
# ---------------------------------------------------------------
class AugmentedDataset(Dataset):
    """
    PyTorch Dataset for loading, augmenting, and pre-processing images and their ground truth masks.

    Args:
        images_dir (str): Path to the directory containing input images.
        groundtruth_dir (str): Path to the directory containing ground truth masks.
        target_size (tuple): Target size for resizing images and masks (width, height).
        transform (callable, optional): Optional transformation to be applied to images and masks.
        threshold (float): Threshold to binarize ground truth masks.

    Attributes:
        image_files (list): Sorted list of image file names.
        groundtruth_files (list): Sorted list of ground truth file names.
    """
    def __init__(self, images_dir, groundtruth_dir, target_size = (416,416),transform=None, threshold = 0.25):
        self.images_dir = images_dir
        self.groundtruth_dir = groundtruth_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))  # Sort to ensure consistent ordering
        self.groundtruth_files = sorted(os.listdir(groundtruth_dir))
        self.threshold = threshold
        self.target_size = target_size
        
        # Debugging/Validation: Print the number of files
        print(f"Number of image files: {len(self.image_files)}")
        print(f"Number of ground truth files: {len(self.groundtruth_files)}")


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and returns one sample (image, ground truth mask) at the given index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            Tuple[Tensor, Tensor]: Normalized image tensor and binarized ground truth tensor.
        """
        # Construct full file paths
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        gt_path = os.path.join(self.groundtruth_dir, self.groundtruth_files[idx])

        # Load image and ground truth mask
        image = Image.open(img_path).convert("RGB")   # Ensure 3-channel RGB image
        groundtruth = Image.open(gt_path).convert("L")  ## Convert ground truth to grayscale
        
        # Resize images and masks to target size using high-quality interpolation
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        groundtruth = groundtruth.resize(self.target_size, Image.Resampling.LANCZOS)

        # Normalize image (scale to range [0, 1])
        image = np.array(image).astype("float32") / 255.0
        
        # Binarize the ground truth mask based on the threshold
        groundtruth = (np.array(groundtruth).astype("float32") / 255.0)
        groundtruth = (groundtruth > self.threshold).astype(np.float32)
        groundtruth = np.expand_dims(groundtruth, axis=-1)

        # Convert to PyTorch tensors and change to (C, H, W) format
        image = torch.from_numpy(image.transpose((2, 0, 1)))  # CxHxW
        groundtruth = torch.from_numpy(groundtruth.transpose((2, 0, 1)))

        return image, groundtruth


# ---------------------------------------------------------------
# TestDataset: Dataset for Test Images Organized in Folders
# ---------------------------------------------------------------
class TestDataset(Dataset):
    """
    PyTorch Dataset for loading test images organized into subdirectories. 
    Each subdirectory is expected to contain one image.

    Args:
        images_dir (str): Path to the directory containing subdirectories with test images.
        transform (callable, optional): Optional transformation to be applied to test images.

    Attributes:
        image_folders (list): Sorted list of subdirectory paths containing test images.
    """
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # Natural sorting function to sort folder names containing numbers
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

        # List all subdirectories (test image folders) and sort them
        self.image_folders = sorted(
            (os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, f))),
            key=natural_sort_key
        )

    def __len__(self):
        """
        Returns the total number of test samples (folders).
        """
        return len(self.image_folders)

    def __getitem__(self, idx):
        """
        Loads and returns one test image from the corresponding folder.

        Args:
            idx (int): Index of the folder to load the image from.

        Returns:
            Tensor: Processed image tensor.
        """
        # Get the folder path for the current index
        folder_path = self.image_folders[idx]
        
        # Find image files (must contain exactly one image file)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(image_files) != 1:
            raise ValueError(f"Expected one image per folder, found {len(image_files)} in {folder_path}")

        # Load the image
        img_path = os.path.join(folder_path, image_files[0])
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB image

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image
