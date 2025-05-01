from helpers import *
import os
import matplotlib.pyplot as plt
from NN.loss import *
from NN.linknet import *
from NN.unet import *
from NN.segnet import *
from dataset_import import *
from dataset_import import TestDataset
import argparse
from tqdm import tqdm

# Dictionary for available models
model_dict = {
    "NL_LinkNet_EGaussian": NL_LinkNet_EGaussian,
    "LinkNet34": LinkNet34,
    "DLinkNet34": DLinkNet34,
    "UNet": UNet,
    "SegNet": SegNet,
}

def test_pred(args):
    """
    Perform mask prediction on test images using selected segmentation model(s).
    Args:
        args: Parsed arguments to specify which models to use.
    """

    # Select device for computation (GPU if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    def load_model(model_class, model_path):
        """
        Load a model from the specified path.
        Args:
            model_class: The class of the model to instantiate.
            model_path (str): Path to the model's checkpoint file.
        Returns:
            model: Loaded model in evaluation mode.
        """
        model = model_class(num_channels=3, num_classes=1).to(device)
        model.load_state_dict(torch.load(model_path))
        return model.eval()

    # Load selected models into a dictionary
    models = {}
    if args.unet:
        models["unet"] = load_model(UNet, args.unet)
    if args.segnet:
        models["segnet"] = load_model(SegNet, args.segnet)
    if args.linknet:
        models["linknet"] = load_model(NL_LinkNet_EGaussian, args.linknet)
    if args.dlinknet:
        models["dlinknet"] = load_model(DLinkNet34, args.dlinknet)

    # Define image transformations for test data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Prepare output directory for saving predicted masks
    if not os.path.exists(args.output_masks):
        os.makedirs(args.output_masks)
    else:
        # Remove any existing content in the output directory
        for file in os.listdir(args.output_masks):
            os.remove(os.path.join(args.output_masks, file))

    # Load test dataset
    test_dataset = TestDataset(args.test_set, transform=transform)

    # Perform prediction for each test image
    for i in tqdm(range(len(test_dataset)), desc="Predicting test images"):
        image = test_dataset[i].unsqueeze(0).to(device)
        outputs = []
        with torch.no_grad():
            for model_name, model in models.items():
                outputs.append(model(image))

        # Average predictions across selected models
        output = torch.mean(torch.stack(outputs), dim=0)
        output = output > 0.5
        output = output.squeeze().cpu().numpy()

        # Save the predicted mask
        plt.imsave(os.path.join(args.output_masks, f"test_{i + 1}_mask.png"), output, cmap="gray")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict segmentation masks for test images using trained model(s)."
    )
    parser.add_argument("--unet", type=str, help="Path to the UNet model checkpoint.")
    parser.add_argument("--segnet", type=str, help="Path to the SegNet model checkpoint.")
    parser.add_argument("--linknet", type=str, help="Path to the LinkNet model checkpoint.")
    parser.add_argument("--dlinknet", type=str, help="Path to the DLinkNet model checkpoint.")
    parser.add_argument("--test_set", type=str, default="dataset/test_set_images",  help="Path to the test dataset.")
    parser.add_argument("--output_masks", type=str, default="test_set_masks", help="Directory to save predicted masks.")

    args = parser.parse_args()

    test_pred(args)
