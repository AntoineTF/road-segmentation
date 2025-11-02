from torchvision.transforms.functional import adjust_brightness, adjust_contrast
import torchvision.transforms.functional as TF
import torch
import math
import PIL.Image as Image
import os

# --- Utility Functions for Image Augmentation ---


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

def pad_image(image, angle):
    """
    Pad the input image to prevent cropping after rotation.
    Args:
        image (PIL.Image): Input image to pad.
        angle (float): Rotation angle in degrees.
    Returns:
        PIL.Image: Padded image.
    """
    original_size = image.size[0]  # Assuming square images
    padding_size = calculate_padding(original_size, angle)
    padded_image = TF.pad(image, padding_size, padding_mode="reflect")
    return padded_image

def generate_augmented_images_with_padding(original_images, ground_truths, num_augmentations=3, angle=45):
    """
    Generate augmented images and ground truths with padding to prevent cropping.
    Args:
        original_images (list): List of PIL.Image objects for input images.
        ground_truths (list): List of PIL.Image objects for ground truth masks.
        num_augmentations (int): Number of augmented versions to generate per image.
        angle (int): Maximum angle for rotation (padding calculated accordingly).
    Returns:
        augmented_images (list): List of augmented images.
        augmented_ground_truths (list): List of augmented ground truth masks.
    """
    augmented_images = []
    augmented_ground_truths = []

    for img, gt in zip(original_images, ground_truths):
        # Pad images to prevent cropping
        img = pad_image(img, angle)
        gt = pad_image(gt, angle)

        for _ in range(num_augmentations):
            # Convert to tensors
            img_tensor = TF.to_tensor(img)
            gt_tensor = TF.to_tensor(gt)

            # Random rotation
            angle = torch.randint(0, 360, (1,)).item()
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

def main(
    dataset_dir="dataset",
    training_images_subdir="training/images",
    training_gt_subdir="training/groundtruth",
    output_aug_images_subdir="augmented/images",
    output_aug_gt_subdir="augmented/groundtruth",
    num_augmentations=3
):
    """
    Generate and save augmented images and ground truths for a given dataset.
    Args:
        dataset_dir (str): Base path to the dataset.
        training_images_subdir (str): Subdirectory for training images.
        training_gt_subdir (str): Subdirectory for ground truth masks.
        output_aug_images_subdir (str): Subdirectory to save augmented images.
        output_aug_gt_subdir (str): Subdirectory to save augmented ground truths.
        num_augmentations (int): Number of augmentations per image.
    """
    training_images_path = os.path.join(dataset_dir, training_images_subdir)
    training_gt_path = os.path.join(dataset_dir, training_gt_subdir)
    output_images_path = os.path.join(dataset_dir, output_aug_images_subdir)
    output_gt_path = os.path.join(dataset_dir, output_aug_gt_subdir)

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_gt_path, exist_ok=True)

    image_files = sorted(os.listdir(training_images_path))
    gt_files = sorted(os.listdir(training_gt_path))

    original_images = [Image.open(os.path.join(training_images_path, file)) for file in image_files]
    ground_truths = [Image.open(os.path.join(training_gt_path, file)) for file in gt_files]

    print(f"Loaded {len(original_images)} images and {len(ground_truths)} ground truth masks.")

    aug_images, aug_ground_truths = generate_augmented_images_with_padding(
        original_images, ground_truths, num_augmentations=num_augmentations, angle=45
    )

    print(f"Saving {len(aug_images)} augmented images and masks...")
    for idx, (aug_img, aug_gt) in enumerate(zip(aug_images, aug_ground_truths)):
        aug_img.save(os.path.join(output_images_path, f"aug_image_rdm_{idx + 1}.png"))
        aug_gt.save(os.path.join(output_gt_path, f"aug_gt_rdm_{idx + 1}.png"))

    print("Augmentation with padding complete!")
        
if __name__ == "__main__":
    main()
    