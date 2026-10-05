"""Evaluation metrics for FreshLens CNN using the deployed decision rule."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from freshlens_ai.constants import CLASSES, FRUITS, STATUSES
from freshlens_ai.models import decode_probabilities


def classification_metrics(truth, prediction, names):
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=np.arange(len(names)),
        zero_division=0,
    )

    return {
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_matrix": confusion_matrix(
            truth,
            prediction,
            labels=np.arange(len(names)),
        ).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(names)
        },
    }


def calculate_metrics(truth, probabilities, group_ids):
    truth = np.asarray(
        truth,
        dtype=np.int64,
    )

    decoded = decode_probabilities(
        probabilities,
    )

    groups = defaultdict(list)

    for correct, group in zip(
        decoded["joint"] == truth,
        group_ids,
    ):
        groups[group].append(
            bool(correct)
        )

    fruit_correct = (
        decoded["fruit"] == truth // 2
    )

    condition_correct = (
        decoded["condition"] == truth % 2
    )

    return {
        "images": len(truth),
        "groups": len(groups),
        "joint": classification_metrics(
            truth,
            decoded["joint"],
            CLASSES,
        ),
        "fruit": classification_metrics(
            truth // 2,
            decoded["fruit"],
            FRUITS,
        ),
        "condition": classification_metrics(
            truth % 2,
            decoded["condition"],
            STATUSES,
        ),
        "condition_when_fruit_correct": {
            "images": int(fruit_correct.sum()),
            "accuracy": (
                float(
                    condition_correct[
                        fruit_correct
                    ].mean()
                )
                if fruit_correct.any()
                else None
            ),
        },
        "mean_group_joint_accuracy": float(
            np.mean(
                [
                    np.mean(values)
                    for values in groups.values()
                ]
            )
        ),
        "raw_8_class_argmax_accuracy": float(
            np.mean(
                decoded["raw_joint_argmax"]
                == truth
            )
        ),
        "definitions": {
            "joint": (
                "Both fruit and condition must be correct "
                "under the deployed prediction rule."
            ),
            "condition": (
                "Condition prediction after selecting a fruit; "
                "includes cases where fruit is wrong."
            ),
            "mean_group_joint_accuracy": (
                "Average of per-group image accuracies, "
                "equal weight per declared group."
            ),
        },
    }


@torch.inference_mode()
def evaluate_model(
    model,
    loader,
    device,
    smoothing=0.05,
):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=smoothing,
        reduction="sum",
    )

    probability_parts = []
    truth_parts = []
    indices_parts = []

    total_loss = 0.0

    for images, targets, indices in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # Preserve legacy behavior: evaluation is float32.
        # AMP is used only during training.
        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        total_loss += float(loss)

        probability_parts.append(
            logits.float()
            .softmax(dim=1)
            .cpu()
            .numpy()
        )

        truth_parts.append(
            targets.cpu().numpy()
        )

        indices_parts.append(
            indices.numpy()
        )

    probabilities = np.concatenate(
        probability_parts
    )

    truth = np.concatenate(
        truth_parts
    )

    indices = np.concatenate(
        indices_parts
    )

    order = np.argsort(
        indices
    )

    if not np.array_equal(
        indices[order],
        np.arange(len(loader.dataset)),
    ):
        raise ValueError(
            "Evaluation must cover the entire selected split exactly once."
        )

    probabilities = probabilities[
        order
    ]

    truth = truth[
        order
    ]

    metrics = calculate_metrics(
        truth,
        probabilities,
        [
            row["group_id"]
            for row in loader.dataset.rows
        ],
    )

    metrics["loss"] = (
        total_loss / len(truth)
    )

    metrics["loss_label_smoothing"] = (
        smoothing
    )

    return metrics, probabilities
