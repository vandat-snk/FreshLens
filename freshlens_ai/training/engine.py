"""One-epoch training engine for FreshLens CNN."""

from __future__ import annotations

import torch

from freshlens_ai.models import training_mode


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    stage,
    accumulation,
    smoothing,
):
    """Train exactly one epoch using the legacy FreshLens update rule."""
    training_mode(
        model,
        stage,
    )

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=smoothing,
        reduction="sum",
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    loss_sum = 0.0
    seen = 0

    for batch, (
        images,
        targets,
        _,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # Preserve the exact legacy normalization for the last,
        # possibly-short gradient accumulation window.
        window_start = (
            (batch // accumulation)
            * accumulation
            * loader.batch_size
        )

        window_samples = min(
            accumulation * loader.batch_size,
            len(loader.dataset) - window_start,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)

            summed_loss = criterion(
                logits,
                targets,
            )

        if not bool(
            torch.isfinite(summed_loss)
        ):
            raise RuntimeError(
                "Non-finite training loss. "
                "Stop and inspect the run."
            )

        scaler.scale(
            summed_loss / window_samples
        ).backward()

        if (
            (batch + 1) % accumulation == 0
            or batch + 1 == len(loader)
        ):
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True,
            )

        loss_sum += float(
            summed_loss.detach()
        )

        seen += len(targets)

        if (
            (batch + 1) % 50 == 0
            or batch + 1 == len(loader)
        ):
            print(
                f"  [TRAIN] batch "
                f"{batch + 1}/{len(loader)} "
                f"| loss {loss_sum / seen:.4f}",
                flush=True,
            )

    return loss_sum / seen
