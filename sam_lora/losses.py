import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def focal_loss(preds, targets, alpha=0.25, gamma=2.0, eps=1e-6):
    """
    Calculate the Focal loss.

    Args:
    preds (tensor): Predicted probabilities of shape (k, 1, w, h) with float values.
    targets (tensor): Ground truth labels of shape (k, 1, w, h) with boolean values.
    alpha (float): Weighting factor for the positive class.
    gamma (float): Focusing parameter.
    eps (float): A small value to avoid log(0).

    Returns:
    float: Focal loss.
    """
    # Convert targets to float and flatten
    targets = targets.float()
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate the cross entropy loss
    ce_loss = F.binary_cross_entropy(preds, targets, reduction='none')

    # Calculate pt: probability of the true class
    pt = torch.where(targets == 1, preds, 1 - preds)

    # Calculate Focal loss
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    return focal_loss.mean()



def cal_dice_loss(preds, targets, eps=1e-6):
    """
    Calculate the Dice loss.

    Args:
    preds (tensor): Predicted masks of shape (k, 1, w, h) with float values.
    targets (tensor): Ground truth masks of shape (k, 1, w, h) with boolean values.
    eps (float): A small value to avoid division by zero.

    Returns:
    float: Dice loss.
    """
    # Convert targets to float type
    targets = targets.float()

    # Calculate intersection and union
    intersection = (preds * targets).sum((2, 3))  # Sum over width and height dimensions
    union = preds.sum((2, 3)) + targets.sum((2, 3))

    # Calculate Dice coefficient
    dice_coeff = (2. * intersection + eps) / (union + eps)

    # Dice Loss is 1 minus the Dice coefficient
    dice_loss = 1 - dice_coeff

    # Return the mean Dice loss over the batch
    return dice_loss.mean(),dice_coeff.mean()

def cal_binary_cross_entropy(preds, targets, eps=1e-6):
    """
    Calculate the Binary Cross-Entropy (BCE) loss.

    Args:
    preds (tensor): Predicted masks of shape (k, 1, w, h) with float values.
    targets (tensor): Ground truth masks of shape (k, 1, w, h) with boolean values.
    eps (float): A small value to stabilize log operation.

    Returns:
    float: Binary Cross-Entropy loss.
    """
    # Convert targets to float type
    targets = targets.float()

    # Apply sigmoid to predictions to get probabilities
    preds = torch.sigmoid(preds)

    # Calculate BCE loss
    bce_loss = F.binary_cross_entropy(preds, targets, reduction='mean')

    return bce_loss
def soft_iou_loss(preds, targets, eps=1e-6):
    """
    Calculate the soft Intersection over Union (IoU) loss.

    Args:
    preds (tensor): Predicted masks of shape (k, 1, w, h) with float values.
    targets (tensor): Ground truth masks of shape (k, 1, w, h) with boolean values.
    eps (float): A small value to avoid division by zero.

    Returns:
    float: Soft IoU loss.
    """
    # Convert targets to float type
    targets = targets.float()

    # Calculate intersection and union
    intersection = (preds * targets).sum((2, 3))  # Sum over width and height dimensions
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - intersection

    # Add a small epsilon to the union to avoid division by zero
    union = union + eps

    # Calculate soft IoU
    iou = intersection / union
    # IoU Loss is the negative log of IoU
    soft_iou_loss = -torch.log(iou)

    # Return the mean IoU loss over the batch
    return soft_iou_loss.mean(),iou

def class_accuracy_loss(predictions, targets):
    """
    Calculate categorical cross-entropy loss for class predictions.

    Args:
        predictions (tensor): Logits from the model of shape [batch_size, num_classes, height, width] or [batch_size, 1, height, width] for binary classification.
        targets (tensor): Ground truth labels of shape [batch_size, 1, height, width] with class indices.

    Returns:
        torch.Tensor: Loss value.
    """
    # Ensure targets are of type torch.long and squeeze the second dimension if it exists
    targets = targets.squeeze(1).float()  # This will change shape from (k, 1, w, h) to (k, w, h)
    predictions = predictions.squeeze(1).float()

    loss = F.cross_entropy(predictions, targets, reduction='mean')
    return loss

def overlay_masks(image_tensor, mask_tensor):
    # image_tensor: torch.Tensor of shape (3, w, h)
    # mask_tensor: torch.Tensor of shape (k, 1, w, h)

    # Check dimensions and squeeze any singleton dimensions in masks
    image_tensor = image_tensor.float() / 255.0  # Normalize image
    mask_tensor = mask_tensor.squeeze(1)  # Remove the singleton dimension

    # Create an RGB image from the original tensor
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Generate a color map with k distinct colors
    cmap = plt.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, mask_tensor.shape[0]))

    # Create an empty canvas for the colored masks
    mask_overlay = np.zeros_like(img)

    # Iterate over each mask and color
    for i, (mask, color) in enumerate(zip(mask_tensor, colors)):
        # Extract the mask and expand it to 3 channels
        mask_expanded = mask.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()

        # Apply color to the mask
        mask_color = mask_expanded * np.array(color[:3])  # Ignore alpha if present

        # Blend colored mask with the previous overlays
        mask_overlay = np.where(mask_expanded, mask_color, mask_overlay)

    # Blend original image with mask overlay
    # Adjust blending factor to change visibility
    alpha = 0.5  # transparency factor
    final_image = (1 - alpha) * img + alpha * mask_overlay

    # Ensure the final image is within correct bounds
    final_image = np.clip(final_image, 0, 1)

    # Convert to image using PIL
    final_image = (final_image * 255).astype(np.uint8)  # Convert to 8-bit per channel
    pil_image = Image.fromarray(final_image)

    return pil_image

