import torch
from torch import nn
from torch.nn import functional as F

TEXT_CLASS: int = 1
SIGNAL_CLASS: int = 2


def soft_erode(prob_map: torch.Tensor) -> torch.Tensor:
    """
    Grayscale morphological erosion of a soft (probability-valued) map, computed as the min of two
    axis-aligned 1D max-pools (erosion is dilation of the negated input, negated back).

    Args:
        prob_map: Tensor of shape (N, C, H, W) with values in [0, 1].

    Returns:
        Eroded tensor of the same shape.
    """
    eroded_rows = -F.max_pool2d(-prob_map, kernel_size=(3, 1), stride=1, padding=(1, 0))
    eroded_cols = -F.max_pool2d(-prob_map, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(eroded_rows, eroded_cols)


def soft_dilate(prob_map: torch.Tensor) -> torch.Tensor:
    """
    Grayscale morphological dilation of a soft (probability-valued) map via a 3x3 max-pool.

    Args:
        prob_map: Tensor of shape (N, C, H, W) with values in [0, 1].

    Returns:
        Dilated tensor of the same shape.
    """
    dilated: torch.Tensor = F.max_pool2d(prob_map, kernel_size=3, stride=1, padding=1)
    return dilated


def soft_open(prob_map: torch.Tensor) -> torch.Tensor:
    """Grayscale morphological opening (erosion followed by dilation)."""
    return soft_dilate(soft_erode(prob_map))


def soft_skeletonize(prob_map: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    Differentiable soft skeletonization, following Shit et al., "clDice - A Novel Topology-Preserving
    Loss Function for Tubular Structure Segmentation" (CVPR 2021). Iteratively erodes the map and keeps
    whatever each erosion step removes that opening would not have removed (i.e. the thin centerline
    parts), accumulating them into a soft skeleton mask.

    Args:
        prob_map: Tensor of shape (N, C, H, W) with values in [0, 1].
        iterations: Number of erosion steps. Should be roughly the radius (in pixels) of the thickest
            structure being skeletonized.

    Returns:
        Soft skeleton tensor of the same shape, with values in [0, 1].
    """
    img = prob_map
    skeleton = F.relu(img - soft_open(img))
    for _ in range(iterations):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cl_dice(pred: torch.Tensor, target: torch.Tensor, iterations: int = 10, smooth: float = 1e-3) -> torch.Tensor:
    """
    Soft clDice loss between predicted and target soft masks, measuring topological/connectivity
    agreement rather than pixel overlap: precision of the predicted skeleton against the target mask,
    and sensitivity of the target skeleton against the predicted mask.

    Args:
        pred: Predicted soft mask, shape (N, C, H, W), values in [0, 1].
        target: Target soft mask, shape (N, C, H, W), values in [0, 1].
        iterations: Number of soft-skeletonization erosion steps.
        smooth: Smoothing term to avoid division by zero.

    Returns:
        Per-(N, C) clDice loss tensor of shape (N, C), where 0 means perfect topological agreement.
    """
    skel_pred = soft_skeletonize(pred, iterations)
    skel_target = soft_skeletonize(target, iterations)

    dims = (2, 3)
    topology_precision = (torch.sum(skel_pred * target, dim=dims) + smooth) / (torch.sum(skel_pred, dim=dims) + smooth)
    topology_sensitivity = (torch.sum(skel_target * pred, dim=dims) + smooth) / (
        torch.sum(skel_target, dim=dims) + smooth
    )

    cl_dice = (2 * topology_precision * topology_sensitivity) / (topology_precision + topology_sensitivity + smooth)
    loss: torch.Tensor = 1 - cl_dice
    return loss


def rgb_to_one_hot(rgb_labels: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB labels to one-hot encoded format.

    Args:
        rgb_labels: Tensor of shape (N, 3, H, W) with RGB labels.

    Returns:
        One-hot encoded tensor of shape (N, 4, H, W)
    """
    labels_one_hot = torch.zeros(
        (rgb_labels.shape[0], 4, rgb_labels.shape[2], rgb_labels.shape[3]), device=rgb_labels.device
    )
    rgb_labels_sum = rgb_labels.sum(dim=1, keepdim=True)
    rgb_labels_sum = torch.where(rgb_labels_sum > 1, rgb_labels_sum, torch.ones_like(rgb_labels_sum))
    rgb_labels = rgb_labels / rgb_labels_sum

    labels_one_hot[:, 0, :, :] = rgb_labels[:, 0, :, :]
    labels_one_hot[:, 1, :, :] = rgb_labels[:, 1, :, :]
    labels_one_hot[:, 2, :, :] = rgb_labels[:, 2, :, :]
    labels_one_hot[:, 3, :, :] = (1 - rgb_labels.sum(dim=1)).clamp(0, 1)  # Background class
    return labels_one_hot


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        signal_class: int = SIGNAL_CLASS,
        union_exponent: int = 2,
        gamma: float = 2.0,
        smooth: float = 1e-3,
        cl_dice_weight: float = 0.0,
        cl_dice_iterations: int = 10,
    ):
        """
        Combined Soft Dice + Focal Loss for multi-class classification, with an optional topology-aware
        clDice term (Shit et al., CVPR 2021) applied to signal_class. ECG traces and grid lines are thin,
        curvilinear structures where pixel-overlap losses (Dice/Focal) can be satisfied by a mask that is
        locally accurate but broken into disconnected fragments; clDice instead rewards predictions whose
        centerline/skeleton matches the target's, directly penalizing gaps and spurious branches.

        Args:
            alpha (float): Weight multiplier for the signal_class in Dice loss.
            signal_class (int): Index of the class to apply extra weight to in Dice loss, and the class
                the clDice term is computed on.
            union_exponent (int): Exponent for the union term in Dice loss (1 or 2).
            gamma (float): Focusing parameter for Focal Loss.
            smooth (float): Small value to avoid division by zero.
            cl_dice_weight (float): Weight of the clDice term added to the total loss. 0.0 (default)
                disables it, leaving the loss identical to plain Dice + Focal.
            cl_dice_iterations (int): Number of soft-skeletonization erosion steps for the clDice term.
                Should be roughly the pixel radius of the thickest traces/lines being segmented.
        """
        super().__init__()
        self.alpha = alpha
        self.signal_class = signal_class
        self.union_exponent = union_exponent
        self.gamma = gamma
        self.smooth = smooth
        self.cl_dice_weight = cl_dice_weight
        self.cl_dice_iterations = cl_dice_iterations

    def forward(self, logits: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (N, C, ...) with raw, unnormalized scores for each class.
            target: Tensor of shape (N, ...) with class indices (0 <= target < C).

        Returns:
            Combined loss scalar.
        """
        target_one_hot = rgb_to_one_hot(target_rgb)  # Convert RGB labels to one-hot encoding
        probs = F.softmax(logits, dim=1)  # (N, C, ...)

        dims = tuple(range(2, logits.dim()))  # spatial dims to sum over

        intersection = torch.sum(probs * target_one_hot, dim=dims)  # (N, C)
        if self.union_exponent == 1:
            union = torch.sum(probs + target_one_hot, dim=dims)  # (N, C)
        elif self.union_exponent == 2:
            union = torch.sum(probs**2 + target_one_hot**2, dim=dims)  # (N, C)
        else:
            raise ValueError("union_exponent must be 1 or 2")

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        weights = torch.ones_like(dice_score)
        weights[:, self.signal_class] = self.alpha

        dice_loss = 1 - dice_score
        weighted_dice_loss = (dice_loss * weights).mean()

        pt = torch.sum(probs * target_one_hot, dim=1)  # (N, ...)
        focal_loss = -((1 - pt) ** self.gamma) * torch.log(pt + self.smooth)
        focal_loss = focal_loss.mean()

        loss: torch.Tensor = weighted_dice_loss + focal_loss

        if self.cl_dice_weight > 0:
            signal_probs = probs[:, [self.signal_class], :, :]
            signal_target = target_one_hot[:, [self.signal_class], :, :]
            cl_dice_loss = soft_cl_dice(signal_probs, signal_target, iterations=self.cl_dice_iterations).mean()
            loss = loss + self.cl_dice_weight * cl_dice_loss

        return loss
