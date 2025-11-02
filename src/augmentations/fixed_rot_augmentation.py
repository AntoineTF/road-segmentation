from torchvision.transforms.functional import adjust_brightness, adjust_contrast
import torchvision.transforms.functional as TF
import torch
import PIL.Image as Image
import os
import math


# --- Utility Functions ---

def calculate_padding(original_size, angle):
    """
    Calculate padding size required to prevent cropping after rotation.
    Args:
        original_size (int): Width or height of the square image.
        angle (float): Rotation angle in degrees.
    Returns:
        int: Padding size in pixels.
    """
    angle_rad = math.radians(angle)
    diagonal = math.sqrt(2) * original_size
    extra_size = diagonal - original_size
    return math.ceil(extra_size / 2)

def pad_image(image, max_angle=270):
    """
    Pad the input image to prevent cropping after rotation.
    Args:
        image (PIL.Image): Input image to pad.
        max_angle (float): Maximum rotation angle in degrees.
    Returns:
        PIL.Image: Padded image.
    """
    original_size = image.size[0]
    padding_size = calculate_padding(original_size, max_angle)
    return TF.pad(image, padding_size, padding_mode="reflect")

def apply_image_only_transform(image_tensor):
    """
    Adjust the brightness and contrast of an image tensor randomly.
    Args:
        image_tensor (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Transformed image tensor with adjusted brightness and contrast.
    """
    image_tensor = adjust_brightness(image_tensor, brightness_factor=0.8 + 0.4 * torch.rand(1).item())  # Range: 0.8 to 1.2
    image_tensor = adjust_contrast(image_tensor, contrast_factor=0.8 + 0.4 * torch.rand(1).item())  # Range: 0.8 to 1.2
    return image_tensor

def add_gaussian_noise(img_tensor, mean=0.0, std=0.08):
    """
    Add Gaussian noise to a tensor image.
    Args:
        img_tensor (torch.Tensor): Input image tensor.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    Returns:
        torch.Tensor: Image tensor with added Gaussian noise.
    """
    noise = torch.randn_like(img_tensor) * std + mean
    return img_tensor + noise

# --- Main Augmentation Logic ---

def generate_augmented_images_with_padding(original_images, ground_truths, rotations, max_angle):
    """
    Generate augmented images and ground truths using padding, fixed rotations, random flips, and Gaussian noise.
    Args:
        original_images (list of PIL.Image): List of original input images.
        ground_truths (list of PIL.Image): Corresponding ground truth images.
        rotations (list of int): List of fixed rotation angles.
        max_angle (int): Maximum angle for padding calculations.
    Returns:
        tuple: Augmented images and augmented ground truth masks.
    """
    augmented_images = []
    augmented_ground_truths = []

    for img, gt in zip(original_images, ground_truths):
        # Pad images to prevent cropping
        img = pad_image(img, max_angle)
        gt = pad_image(gt, max_angle)

        for angle in rotations:
            # Convert to tensors
            img_tensor = TF.to_tensor(img)
            gt_tensor = TF.to_tensor(gt)

            # Apply fixed rotation
            img_tensor = TF.rotate(img_tensor, angle)
            gt_tensor = TF.rotate(gt_tensor, angle)

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                gt_tensor = TF.hflip(gt_tensor)

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                gt_tensor = TF.vflip(gt_tensor)

            # Add Gaussian noise to the image
            img_tensor = add_gaussian_noise(img_tensor)

            # Apply additional color transformations only to the image
            img_tensor = apply_image_only_transform(img_tensor)

            # Convert back to PIL images
            augmented_images.append(TF.to_pil_image(img_tensor.clamp(0, 1)))
            augmented_ground_truths.append(TF.to_pil_image(gt_tensor))

    return augmented_images, augmented_ground_truths

def save_augmented_data(aug_images, aug_ground_truths, output_images_path, output_gt_path):
    """
    Save augmented images and ground truths to specified directories.
    Args:
        aug_images (list): List of augmented images.
        aug_ground_truths (list): List of augmented ground truth masks.
        output_images_path (str): Path to save augmented images.
        output_gt_path (str): Path to save augmented ground truths.
    """
    for idx, (aug_img, aug_gt) in enumerate(zip(aug_images, aug_ground_truths)):
        aug_img.save(os.path.join(output_images_path, f"aug_image_fx_{idx + 1}.png"))
        aug_gt.save(os.path.join(output_gt_path, f"aug_gt_fx_{idx + 1}.png"))

# --- Main Function ---

def main(
    dataset_dir="dataset",
    training_images_subdir="training/images",
    training_gt_subdir="training/groundtruth",
    output_aug_images_subdir="augmented_dataset/images",
    output_aug_gt_subdir="augmented_dataset/groundtruth",
    rotations=[15, 30, 45, 60, 90, 180, 270],
    max_angle=270,
    num_images=100
):
    """
    Generate and save augmented images and corresponding ground truths using padding, fixed rotations, and random flips.
    Args:
        dataset_dir (str): Base path to the dataset.
        training_images_subdir (str): Subdirectory containing training images.
        training_gt_subdir (str): Subdirectory containing ground truth masks.
        output_aug_images_subdir (str): Subdirectory for saving augmented images.
        output_aug_gt_subdir (str): Subdirectory for saving augmented ground truths.
        rotations (list of int): List of rotation angles to apply.
        max_angle (int): Maximum rotation angle for padding calculations.
        num_images (int): Number of images to process (assumes sequential naming).
    """
    training_images_path = os.path.join(dataset_dir, training_images_subdir)
    training_gt_path = os.path.join(dataset_dir, training_gt_subdir)
    output_images_path = os.path.join(dataset_dir, output_aug_images_subdir)
    output_gt_path = os.path.join(dataset_dir, output_aug_gt_subdir)

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_gt_path, exist_ok=True)

    # Load original images and ground truths
    original_images = [
        Image.open(os.path.join(training_images_path, f"padded_image_{i}.png")) for i in range(1, num_images + 1)
    ]
    ground_truths = [
        Image.open(os.path.join(training_gt_path, f"padded_gt_{i}.png")) for i in range(1, num_images + 1)
    ]

    print(f"Loaded {len(original_images)} images and {len(ground_truths)} ground truth masks.")

    # Generate augmented images
    aug_images, aug_ground_truths = generate_augmented_images_with_padding(
        original_images, ground_truths, rotations=rotations, max_angle=max_angle
    )

    # Save augmented images and ground truths
    save_augmented_data(aug_images, aug_ground_truths, output_images_path, output_gt_path)
    print("Augmentation with fixed rotations and padding completed!")

    
if __name__ == "__main__":
    main()