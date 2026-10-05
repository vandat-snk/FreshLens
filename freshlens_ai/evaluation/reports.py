"""Prediction exports and plots for FreshLens CNN evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from freshlens_ai.constants import CLASSES, FRUITS, STATUSES
from freshlens_ai.models import decode_probabilities
from freshlens_ai.utils.file_io import write_csv


def save_predictions(
    directory,
    prefix,
    rows,
    probabilities,
):
    decoded = decode_probabilities(
        probabilities,
    )

    output = []

    for i, row in enumerate(rows):
        fruit = int(
            decoded["fruit"][i]
        )

        condition = int(
            decoded["condition"][i]
        )

        output.append(
            {
                "path": row["path"],
                "group_id": row["group_id"],
                "true_fruit": row["fruit"],
                "true_condition": row["status"],
                "predicted_fruit": FRUITS[fruit],
                "predicted_condition": STATUSES[condition],
                "joint_correct": (
                    int(decoded["joint"][i])
                    == row["target"]
                ),
                "fruit_score": float(
                    decoded["fruit_probabilities"][
                        i,
                        fruit,
                    ]
                ),
                "condition_score_given_fruit": float(
                    decoded["condition_probabilities"][
                        i,
                        condition,
                    ]
                ),
                **{
                    name + "_score": float(
                        probabilities[i, j]
                    )
                    for j, name in enumerate(CLASSES)
                },
            }
        )

    columns = list(
        output[0]
    )

    write_csv(
        Path(directory)
        / f"{prefix}_predictions.csv",
        output,
        columns,
    )

    write_csv(
        Path(directory)
        / f"{prefix}_errors.csv",
        [
            row
            for row in output
            if not row["joint_correct"]
        ],
        columns,
    )


def plot_confusion(
    path,
    metrics,
    title,
):
    import matplotlib

    matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    values = np.asarray(
        metrics["joint"][
            "confusion_matrix"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(9, 8),
        layout="constrained",
    )

    drawn = axis.imshow(
        values,
        cmap="Blues",
    )

    axis.set(
        xticks=np.arange(8),
        yticks=np.arange(8),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted fruit + condition",
        ylabel="True fruit + condition",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
    )

    for y in range(8):
        for x in range(8):
            axis.text(
                x,
                y,
                str(values[y, x]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if values[y, x]
                    > values.max() * 0.55
                    else "black"
                ),
            )

    figure.colorbar(
        drawn,
        ax=axis,
        label="Images",
        shrink=0.8,
    )

    figure.savefig(
        path,
        dpi=150,
    )

    plt.close(
        figure
    )


def plot_history(
    path,
    history,
):
    import matplotlib

    matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        layout="constrained",
    )

    epochs = [
        row["epoch"]
        for row in history
    ]

    axes[0].plot(
        epochs,
        [
            row["train_loss"]
            for row in history
        ],
        label="Train",
    )

    axes[0].plot(
        epochs,
        [
            row["val_loss"]
            for row in history
        ],
        label="Validation",
    )

    axes[0].set(
        title=(
            "Cross-entropy "
            "(same label smoothing)"
        ),
        xlabel="Epoch",
        ylabel="Loss",
    )

    axes[1].plot(
        epochs,
        [
            row["val_fruit_accuracy"]
            for row in history
        ],
        label="Fruit accuracy",
    )

    axes[1].plot(
        epochs,
        [
            row["val_joint_accuracy"]
            for row in history
        ],
        label="Joint accuracy",
    )

    axes[1].plot(
        epochs,
        [
            row["val_joint_macro_f1"]
            for row in history
        ],
        label="Joint macro-F1",
    )

    axes[1].set(
        title="Validation only",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, 1.02),
    )

    for axis in axes:
        axis.legend()
        axis.grid(
            alpha=0.2
        )

    figure.savefig(
        path,
        dpi=150,
    )

    plt.close(
        figure
    )
