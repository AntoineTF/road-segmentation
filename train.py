from helpers import *
import os
import numpy as np
import matplotlib.pyplot as plt
from NN.loss import * #Import CustomLoss a class where different types of losses are implemented
from NN.linknet import * #Import different linknet models
from NN.unet import * #Import Unet class
from NN.segnet import * #Import SegNet class
import time
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
from dataset_import import *  # Import custom dataset classes

THRESHOLD = 0.25  # Threshold for converting predictions to binary values

def train(model, batch_size=8, epochs=50, lr=1e-4, patience = 5,loss_name="squared_dice", debugging = False):
    """
    Train a segmentation model on the given dataset.

    Args:
        model (str): Name of the model to train (e.g., UNet, LinkNet34, SegNet).
        batch_size (int): Number of samples per batch.
        epochs (int): Total number of training epochs.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        loss_name (str): Name of the loss function to use.
        debugging (bool): Whether to print debugging information for a sample batch.

    Returns:
        model: Trained model.
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
    """   
    
    # Select computation device: GPU (if available) or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Using device: {}".format(device))
    
    # Define model save path
    savepath = "models"
    model_name = "trained_model_" + str(model) +"_" + loss_name + "_"+ str(lr) + "_"+ str(epochs) + str(lr) + "_"+ str(batch_size) +".pt"

    
    # Set target input size based on model type
    target_size = (512, 512) if model in ["SegNet"] else (416, 416)
    
    # Load training dataset
    train_dataset = AugmentedDataset(
        images_dir="mc_dataset/final_dataset/training/images",
        groundtruth_dir="mc_dataset/final_dataset/training/groundtruth",
        target_size = target_size,
        threshold = THRESHOLD,
    )
    print("length of the training dataset :", len(train_dataset))
    
    # Load validation dataset
    val_dataset = AugmentedDataset(
        images_dir="mc_dataset/final_dataset/validation/images",
        groundtruth_dir="mc_dataset/final_dataset/validation/groundtruth",
        target_size = target_size,
        threshold = THRESHOLD,
    )
    print("length of the validation dataset :", len(val_dataset))
    
    # Debugging: Check the shape and range of a sample batch
    if debugging:
        train_image, train_groundtruth = train_dataset[0]
        print("Training image size:", train_image.size())
        print("Training groundtruth size:", train_groundtruth.size())
        print("Image value range:", train_image.min().item(), "to", train_image.max().item())
        print("Groundtruth value range:", train_groundtruth.min().item(), "to", train_groundtruth.max().item())

    
    # DataLoader for batch processing    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define model dictionary
    model_dict = {
    "NL_LinkNet_EGaussian": NL_LinkNet_EGaussian,
    "LinkNet34": LinkNet34,
    "DLinkNet34": DLinkNet34,
    "UNet": UNet,
    "SegNet": SegNet,
    }
    
   # Set sigmoid activation based on loss function
    sigmoid_bool = loss_name not in ["bce"]
    
    # Initialize the model
    ModelClass = model_dict[model]
    model = ModelClass(num_classes=1, sigmoid_bool=sigmoid_bool).to(device)


    # Define optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    calc_loss = Losses(beta=0.8) #alpha and gamma can also be precised, according to the loss used

    # Initialize variables for tracking progress
    best_f1_score = 0.0
    train_losses = []
    val_losses = []
    f1_scores = []
    val_labels_all, val_preds_all = [], []
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        print("-" * 25, "Epoch {}/{}\n".format(epoch, epochs - 1))
        since = time.time()
       
        # Training phase
        model.train()
        train_loss = 0.0
        train_total_samples = 0

        for inputs, labels in tqdm(train_dataloader, desc="Processing Training Data"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = calc_loss(labels,outputs, loss_name)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_total_samples += inputs.size(0)

        # Calculate average training loss
        train_epoch_loss = train_loss / train_total_samples
        train_losses.append(train_epoch_loss)
        print("Training Loss: {:.4f}".format(train_epoch_loss))


        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_preds = []
        val_targets = []
        for inputs, labels in tqdm(val_dataloader, desc="Processing Validation Data"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = calc_loss(labels, outputs, loss_name)
                if not sigmoid_bool:
                    outputs = torch.sigmoid(outputs)
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                val_preds.append(outputs > THRESHOLD)  # Threshold predictions
                val_targets.append(labels > THRESHOLD)

        # Calculate average validation loss and F1 score
        val_epoch_loss = val_loss / val_samples
        val_losses.append(val_epoch_loss)
        val_preds = torch.cat(val_preds).view(-1).cpu().numpy()
        val_targets = torch.cat(val_targets).view(-1).cpu().numpy()
        val_f1_score = f1_score(val_targets, val_preds)
        f1_scores.append(val_f1_score)

        metrics = compute_metrics(val_preds, val_targets, threshold=THRESHOLD)

        
        print("IoU score: {:.4f}".format(compute_IoU(val_preds, val_targets)))
        print("F1 score: {:.4f}".format(val_f1_score))
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        val_labels_all.extend(val_targets)
        val_preds_all.extend(val_preds)
        #scheduler.step()
        print("Validation Loss: {:.4f}".format(val_epoch_loss))

        # Save the best model
        if best_f1_score < val_f1_score:
            best_f1_score = val_f1_score
            best_epoch = epoch
            patience_counter = 0
            save_model(model, savepath=savepath, model_name=model_name)
            print(
                "New best model {} saved with f1 score: {:.4f}".format(
                    os.path.join(savepath, model_name), best_f1_score
                )
            )
        else :
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
        
        if patience_counter > patience:
            print("Early stopping triggered !")
            break
        
        scheduler.step()
        time_elapsed = time.time() - since
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    # save training and validation losses and f1 scores to csv file
    save_losses(
        train_losses,
        val_losses,
        f1_scores,
        savepath=os.path.join(savepath, model_name[:-3]),
    )  # remove .pt from model_name
    print(f"Training stopped at epoch {epoch}, best F1 score: {best_f1_score:.4f}")
    return model, train_losses, val_losses

def main():
    """
    Main function to parse arguments and execute the training process.
    """
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train.") #UNet, LinkNet34, DLinkNet34, SegNet
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--loss_name", type=str, default="squared_dice", help="Loss function to use.")
    args = parser.parse_args()

    # Call the train function with parsed arguments
    train(
        model=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        loss_name=args.loss_name
    )
    print("Training complete!")

    # Plot training and validation losses
    plt.figure(figsize=(5, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()
    
    # Save plot
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    
    plot_filename = os.path.join(
        plots_dir, f"segnet_loss_plot_bce_bs{args.batch_size}_ep{args.epochs}_lr{args.learning_rate}_loss_{args.loss_name}.png"
    )
    plt.savefig(plot_filename)  # Save the plot as a PNG file
    print(f"Plot saved as {plot_filename}")


if __name__ == "__main__":
    main()
