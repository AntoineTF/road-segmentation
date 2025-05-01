# All the losses in this code are derived from the implementations described in the following paper:
# https://doi.org/10.1016/j.jag.2022.103159

import torch
import torch.nn as nn
import torch.nn.functional as F


class Losses(nn.Module):
    """
    A class containing various loss function implementations for binary segmentation tasks.
    The class supports distribution-based, region-based, and compound loss functions.
    """
    def __init__(self, beta=1.0, alpha=0.5, gamma=2.0):
        """
        Initialize the Losses class with parameters for weighted and focal losses.

        Args:
            beta (float): Weight factor for positive class in weighted losses.
            alpha (float): Balance factor for false positives/negatives in Tversky loss.
            gamma (float): Focusing parameter for focal and focal Tversky losses.
        """
        
        super(Losses, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

    ### Distribution-based loss functions
    
    def binary_cross_entropy(self, y_true, y_pred, reduction="mean", weight=None):
        """
        Binary Cross-Entropy (BCE) loss.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.
            reduction (str): Reduction method ('mean', 'sum', 'none').
            weight (Tensor, optional): Weight tensor for balancing the loss.

        Returns:
            Tensor: Computed BCE loss.
        """
        loss = nn.BCEWithLogitsLoss(reduction=reduction, weight=weight)
        return loss(y_pred, y_true)

    def weighted_cross_entropy(self, y_true, y_pred):
        """
        Weighted Binary Cross-Entropy loss with custom positive class weight (beta).

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.

        Returns:
            Tensor: Computed weighted BCE loss.
        """
        weights = self.beta * y_true + (1 - y_true)
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")  
        per_pixel_loss = bce_loss(y_pred, y_true)  
        weighted_loss = per_pixel_loss * weights 
        return weighted_loss.mean()

    def balanced_cross_entropy(self, y_true, y_pred):
        """
        Balanced Binary Cross-Entropy loss.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.

        Returns:
            Tensor: Computed balanced BCE loss.
        """
        weights = self.beta * y_true + (1 - self.beta) * (1 - y_true)
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")  
        per_pixel_loss = bce_loss(y_pred, y_true)  
        weighted_loss = per_pixel_loss * weights 
        return weighted_loss.mean()
    
    def focal_loss(self, y_true, y_pred):
        """
        Focal Loss to address class imbalance by focusing on hard examples.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.

        Returns:
            Tensor: Computed Focal Loss.
        """
        p = torch.sigmoid(y_pred)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        
        p_t = p * y_true + (1 - p) * (1 - y_true)
        modulating_factor = (1 - p_t) ** self.gamma
        
        loss = modulating_factor * bce_loss
        
        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        return loss.mean()


    ### Region-based loss functions
    
    def jaccard_loss(self, y_true, y_pred, smooth=1):
        """
        Jaccard loss, also known as Intersection over Union (IoU) loss.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted binary mask.
            smooth (float): Smoothing parameter to avoid division by zero.

        Returns:
            Tensor: Computed Jaccard loss.
        """
        inter = (y_true * y_pred).sum()
        total = y_true.sum() + y_pred.sum()
        union = total - inter
        return 1 - (inter + smooth) / (union + smooth)

    def dice_loss(self, y_true, y_pred, smooth=1):
        """
        Dice Loss for segmentation tasks.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted binary mask.
            smooth (float): Smoothing parameter.

        Returns:
            Tensor: Computed Dice loss.
        """
        inter = (y_true * y_pred).sum()
        output_tf = 1 - (2.0 * inter + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        return output_tf

    def squared_dice_loss(self, y_true, y_pred, smooth=1):
        """
        Squared Dice Loss to penalize large errors more.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted binary mask.
            smooth (float): Smoothing parameter.

        Returns:
            Tensor: Computed Squared Dice loss.
        """
        inter = (y_true * y_pred).sum()
        output_tf = 1 - (2.0 * inter + smooth) / (torch.pow(y_true, 2).sum() + torch.pow(y_pred, 2).sum() + smooth)
        return output_tf

    def log_cosh_dice_loss(self, y_true, y_pred):
        """
        Log-Cosh Dice Loss to smooth the Dice loss function.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted binary mask.

        Returns:
            Tensor: Computed Log-Cosh Dice loss.
        """
        x = self.dice_loss(y_true, y_pred)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)

    def tversky_loss(self, y_true, y_pred, smooth=1):
        """
        Tversky Loss to balance false positives and false negatives.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted binary mask.
            smooth (float): Smoothing parameter.

        Returns:
            Tensor: Computed Tversky loss.
        """
        inter = (y_true * y_pred).sum()
        false_negatives = (y_true * (1 - y_pred)).sum()
        false_positives = ((1 - y_true) * y_pred).sum()
        output_tf = 1 - (inter + smooth) / (inter+ self.alpha * false_negatives+ (1 - self.alpha) * false_positives + smooth)
        return output_tf

    def focal_tversky_loss(self, y_true, y_pred, smooth=1):
        """
        Focal Tversky Loss to focus on hard examples.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted binary mask.
            smooth (float): Smoothing parameter.

        Returns:
            Tensor: Computed Focal Tversky loss.
        """
        tversky = self.tversky_loss(y_true, y_pred, smooth)
        output_tf = torch.pow((1 - tversky), self.gamma)
        return output_tf
    
    
    ### Compound loss functions
    
    def bce_dice_loss(self, y_true, y_pred):
        """
        Combined BCE and Dice Loss.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.

        Returns:
            Tensor: Combined BCE and Dice loss.
        """
        output_tf = 0.3 * self.binary_cross_entropy(y_true, y_pred) + 0.7 * self.dice_loss(y_true, y_pred)
        return output_tf

    def combo_loss(self, y_true, y_pred):
        """
        Combination of Weighted BCE and Dice Loss.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.

        Returns:
            Tensor: Combined loss.
        """
        output_tf = self.weighted_cross_entropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)

        return output_tf

    def forward(self, y_true, y_pred, loss_type):
        """
        Compute the loss based on the specified loss type.

        Args:
            y_true (Tensor): Ground truth binary mask.
            y_pred (Tensor): Predicted logits.
            loss_type (str): Type of loss to compute.

        Returns:
            Tensor: Computed loss value.

        Raises:
            ValueError: If an unsupported loss type is specified.
        """
        # Dictionary mapping loss types to their respective functions
        loss_functions = {
            "bce": self.binary_cross_entropy,
            "weighted_bce": self.weighted_cross_entropy,
            "balanced_bce": self.balanced_cross_entropy,
            "focal": self.focal_loss,
            "jaccard": self.jaccard_loss,
            "dice": self.dice_loss,
            "squared_dice": self.squared_dice_loss,
            "log_cosh_dice": self.log_cosh_dice_loss,
            "tversky": self.tversky_loss,
            "focal_tversky": self.focal_tversky_loss,
            "bce_dice": self.bce_dice_loss,
            "combo": self.combo_loss,
        }

        # Validate loss type
        if loss_type not in loss_functions:
            raise ValueError(
                f"Invalid loss type specified: {loss_type}. "
                f"Supported loss types are: {list(loss_functions.keys())}"
            )

        # Call the appropriate loss function
        return loss_functions[loss_type](y_true, y_pred)