import torch

from src.loss.loss import SIGNAL_CLASS, DiceFocalLoss, soft_cl_dice


def _make_line(height: int, width: int, row: int, col_start: int, col_end: int) -> torch.Tensor:
    """Creates a (1, 1, height, width) soft mask with a 1px-thick horizontal line segment set to 1."""
    mask = torch.zeros(1, 1, height, width)
    mask[0, 0, row, col_start:col_end] = 1.0
    return mask


def _make_band(height: int, width: int, row_start: int, row_end: int, col_start: int, col_end: int) -> torch.Tensor:
    """Creates a (1, 1, height, width) soft mask with a filled rectangular band set to 1."""
    mask = torch.zeros(1, 1, height, width)
    mask[0, 0, row_start:row_end, col_start:col_end] = 1.0
    return mask


def test_soft_cl_dice_perfect_match_is_near_zero() -> None:
    line = _make_line(32, 32, 16, 5, 25)
    loss = soft_cl_dice(line, line, iterations=3)
    assert loss.shape == (1, 1)
    assert loss.item() < 1e-2


def test_soft_cl_dice_disjoint_masks_is_near_max() -> None:
    target = _make_line(32, 32, 8, 5, 25)
    pred = _make_line(32, 32, 24, 5, 25)  # Non-overlapping row.
    loss = soft_cl_dice(pred, target, iterations=3)
    assert loss.item() > 0.9


def test_soft_cl_dice_penalizes_broken_connectivity_more_than_thinning() -> None:
    # A thin (1px) line has no thickness for the soft-skeleton to erode away, so its skeleton is just
    # itself, which makes clDice degenerate to a plain overlap ratio. A band with real thickness (like an
    # ECG trace stroke) is needed for skeletonization -- and hence clDice's topology-awareness -- to do
    # anything: eroding down to a 1px centerline is what lets a mid-line cut create two new skeleton
    # endpoints (extra erosion loss) instead of just retracting the two pre-existing ends.
    target = _make_band(32, 32, 14, 19, 5, 25)

    # Same band, shortened by 1 column on each end. Still a single connected segment.
    thinned = _make_band(32, 32, 14, 19, 6, 24)

    # Same band, but with a gap punched through the middle, splitting it into two disconnected segments.
    # Removes the same number of pixels as the thinning case above, so plain pixel-overlap loss is
    # comparable, but topology (connectivity) differs.
    broken = target.clone()
    broken[0, 0, 14:19, 14:16] = 0.0

    thinned_loss = soft_cl_dice(thinned, target, iterations=4).item()
    broken_loss = soft_cl_dice(broken, target, iterations=4).item()

    assert broken_loss > thinned_loss


def test_soft_cl_dice_is_differentiable() -> None:
    target = _make_line(16, 16, 8, 2, 14)
    pred = _make_line(16, 16, 8, 2, 14).clone() * 0.6
    pred.requires_grad_(True)

    loss = soft_cl_dice(pred, target, iterations=2).sum()
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def _make_logits_and_target(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    target_rgb = torch.zeros(1, 3, height, width)
    target_rgb[0, SIGNAL_CLASS, height // 2, 4 : width - 4] = 1.0

    logits = torch.randn(1, 4, height, width) * 0.1
    logits[0, SIGNAL_CLASS, height // 2, 4 : width - 4] = 2.0  # Roughly correct, imperfect prediction.
    return logits, target_rgb


def test_dice_focal_loss_default_disables_cl_dice_term() -> None:
    logits, target_rgb = _make_logits_and_target(24, 24)

    loss_without_kwarg = DiceFocalLoss()(logits, target_rgb)
    loss_with_zero_weight = DiceFocalLoss(cl_dice_weight=0.0)(logits, target_rgb)

    assert torch.isclose(loss_without_kwarg, loss_with_zero_weight)


def test_dice_focal_loss_cl_dice_weight_changes_loss() -> None:
    logits, target_rgb = _make_logits_and_target(24, 24)

    baseline = DiceFocalLoss(cl_dice_weight=0.0)(logits, target_rgb)
    with_topology = DiceFocalLoss(cl_dice_weight=0.5, cl_dice_iterations=3)(logits, target_rgb)

    assert not torch.isclose(baseline, with_topology)
    assert torch.isfinite(with_topology)
