"""Metrics and inference evaluation share the exact deployed decision rule."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from cnn_data import CLASSES, FRUITS, STATUSES, atomic_json, write_csv
from cnn_model import decode_probabilities


def classification_metrics(truth, prediction, names):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(len(names)), zero_division=0)
    return {
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(len(names))).tolist(),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(names)},
    }


def calculate_metrics(truth, probabilities, group_ids):
    truth = np.asarray(truth, dtype=np.int64)
    decoded = decode_probabilities(probabilities)
    groups = defaultdict(list)
    for correct, group in zip(decoded["joint"] == truth, group_ids):
        groups[group].append(bool(correct))
    fruit_correct = decoded["fruit"] == truth // 2
    condition_correct = decoded["condition"] == truth % 2
    return {
        "images": len(truth), "groups": len(groups),
        "joint": classification_metrics(truth, decoded["joint"], CLASSES),
        "fruit": classification_metrics(truth // 2, decoded["fruit"], FRUITS),
        "condition": classification_metrics(truth % 2, decoded["condition"], STATUSES),
        "condition_when_fruit_correct": {
            "images": int(fruit_correct.sum()),
            "accuracy": float(condition_correct[fruit_correct].mean()) if fruit_correct.any() else None,
        },
        "mean_group_joint_accuracy": float(np.mean([np.mean(values) for values in groups.values()])),
        "raw_8_class_argmax_accuracy": float(np.mean(decoded["raw_joint_argmax"] == truth)),
        "definitions": {
            "joint": "Both fruit and condition must be correct under the deployed prediction rule.",
            "condition": "Condition prediction after selecting a fruit; includes cases where fruit is wrong.",
            "mean_group_joint_accuracy": "Average of per-group image accuracies, equal weight per declared group.",
        },
    }


@torch.inference_mode()
def evaluate_model(model, loader, device, smoothing=.05):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="sum")
    probability_parts, truth_parts, indices_parts = [], [], []
    total_loss = 0.
    for images, targets, indices in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # Evaluate in float32, as in PREDICT_CNN; AMP is only for training.
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss)
        probability_parts.append(logits.float().softmax(dim=1).cpu().numpy())
        truth_parts.append(targets.cpu().numpy())
        indices_parts.append(indices.numpy())
    probabilities = np.concatenate(probability_parts)
    truth = np.concatenate(truth_parts)
    indices = np.concatenate(indices_parts)
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(len(loader.dataset))):
        raise ValueError("Evaluation must cover the entire selected split exactly once.")
    probabilities, truth = probabilities[order], truth[order]
    metrics = calculate_metrics(truth, probabilities, [row["group_id"] for row in loader.dataset.rows])
    metrics["loss"] = total_loss / len(truth)
    metrics["loss_label_smoothing"] = smoothing
    return metrics, probabilities


def save_predictions(directory, prefix, rows, probabilities):
    decoded = decode_probabilities(probabilities)
    output = []
    for i, row in enumerate(rows):
        fruit, condition = int(decoded["fruit"][i]), int(decoded["condition"][i])
        output.append({
            "path": row["path"], "group_id": row["group_id"],
            "true_fruit": row["fruit"], "true_condition": row["status"],
            "predicted_fruit": FRUITS[fruit], "predicted_condition": STATUSES[condition],
            "joint_correct": int(decoded["joint"][i]) == row["target"],
            "fruit_score": float(decoded["fruit_probabilities"][i, fruit]),
            "condition_score_given_fruit": float(decoded["condition_probabilities"][i, condition]),
            **{name + "_score": float(probabilities[i, j]) for j, name in enumerate(CLASSES)},
        })
    columns = list(output[0])
    write_csv(Path(directory) / f"{prefix}_predictions.csv", output, columns)
    write_csv(Path(directory) / f"{prefix}_errors.csv", [row for row in output if not row["joint_correct"]], columns)


def plot_confusion(path, metrics, title):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    values = np.asarray(metrics["joint"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    drawn = axis.imshow(values, cmap="Blues")
    axis.set(xticks=np.arange(8), yticks=np.arange(8), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted fruit + condition", ylabel="True fruit + condition", title=title)
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    for y in range(8):
        for x in range(8):
            axis.text(x, y, str(values[y, x]), ha="center", va="center",
                      color="white" if values[y, x] > values.max() * .55 else "black")
    figure.colorbar(drawn, ax=axis, label="Images", shrink=.8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_history(path, history):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Cross-entropy (same label smoothing)", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["val_fruit_accuracy"] for row in history], label="Fruit accuracy")
    axes[1].plot(epochs, [row["val_joint_accuracy"] for row in history], label="Joint accuracy")
    axes[1].plot(epochs, [row["val_joint_macro_f1"] for row in history], label="Joint macro-F1")
    axes[1].set(title="Validation only", xlabel="Epoch", ylabel="Score", ylim=(0, 1.02))
    for axis in axes:
        axis.legend(); axis.grid(alpha=.2)
    figure.savefig(path, dpi=150)
    plt.close(figure)
