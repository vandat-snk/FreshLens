"""Two-stage fine-tuning; only train/validation images enter this process."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from cnn_data import PROJECT_DIR, DataError, FruitDataset, atomic_json, file_sha, load_locked_dataset, verify_images, write_csv
from cnn_model import checkpoint_metadata, choose_device, configure_stage, make_model, read_checkpoint, save_checkpoint, training_mode
from cnn_metrics import evaluate_model, plot_confusion, plot_history, save_predictions


def optimizer_for(model, stage):
    configure_stage(model, stage)
    if stage == "warmup":
        return torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    return torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)


def train_epoch(model, loader, optimizer, scaler, device, stage, accumulation, smoothing):
    training_mode(model, stage)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="sum")
    optimizer.zero_grad(set_to_none=True)
    loss_sum, seen = 0., 0
    for batch, (images, targets, _) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # Correct normalization even when the final accumulation window/batch is short.
        window_start = (batch // accumulation) * accumulation * loader.batch_size
        window_samples = min(accumulation * loader.batch_size, len(loader.dataset) - window_start)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            summed_loss = criterion(logits, targets)
        if not bool(torch.isfinite(summed_loss)):
            raise RuntimeError("Non-finite training loss. Stop and inspect the run.")
        scaler.scale(summed_loss / window_samples).backward()
        if (batch + 1) % accumulation == 0 or batch + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(summed_loss.detach())
        seen += len(targets)
        if (batch + 1) % 50 == 0 or batch + 1 == len(loader):
            print(f"  [TRAIN] batch {batch + 1}/{len(loader)} | loss {loss_sum / seen:.4f}", flush=True)
    return loss_sum / seen


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(args):
    if min(args.batch_size, args.accumulation, args.warmup_epochs, args.finetune_epochs, args.patience) < 1:
        raise DataError("Batch size, accumulation, epochs and patience must be positive.")
    if args.workers < 0 or not 0 <= args.label_smoothing < 1:
        raise DataError("Invalid workers or label smoothing.")
    device = choose_device(args.device)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise DataError("Output already exists. Use --resume for this run, or select a NEW --output directory.")
    rows, identity = load_locked_dataset(args.data)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    config = {
        "dataset_root": str(Path(args.root).resolve()), "dataset_dir": str(Path(args.data).resolve()),
        "batch_size": args.batch_size, "accumulation": args.accumulation,
        "warmup_epochs": args.warmup_epochs, "finetune_epochs": args.finetune_epochs,
        "patience": args.patience, "seed": args.seed, "workers": args.workers,
        "label_smoothing": args.label_smoothing, "pretrained": not args.from_scratch,
        "device": args.device,
    }
    restored = read_checkpoint(output / "last.pt") if args.resume else None
    if restored:
        if restored.get("dataset_identity") != identity or restored.get("config") != config:
            raise DataError("Resume dataset/config differs from last.pt. Restore the original command/config.")
        if not (output / "best.pt").is_file():
            raise DataError("The best.pt belonging to this run is missing.")
    print(f"[DATA] train={len(train_rows)}; val={len(val_rows)}; test={identity['split_counts']['test']} reserved", flush=True)
    print("[DATA] Only train and validation image files will be read.", flush=True)
    verify_images(args.root, train_rows + val_rows)
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    training = FruitDataset(args.root, train_rows, training=True)
    validation = FruitDataset(args.root, val_rows, training=False)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    train_loader = DataLoader(training, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(validation, shuffle=False, **loader_options)
    print("[MODEL] Loading EfficientNet-B0" + (" pretrained ImageNet weights..." if not restored and not args.from_scratch else "..."), flush=True)
    model = make_model(pretrained=not args.from_scratch and not restored, cache_dir=PROJECT_DIR / ".cache" / "torch").to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_score, best_loss, bad_epochs = [], -1., float("inf"), 0
    optimizer, scheduler, active_stage = None, None, None
    start_epoch = 0
    best_metrics = None
    if restored:
        model.load_state_dict(restored["model_state"], strict=True)
        active_stage = restored["stage"]
        optimizer = optimizer_for(model, active_stage)
        optimizer.load_state_dict(restored["optimizer_state"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-6) if active_stage == "finetune" else None
        if scheduler:
            scheduler.load_state_dict(restored["scheduler_state"])
        scaler.load_state_dict(restored["scaler_state"])
        history, start_epoch, bad_epochs = restored["history"], restored["epoch"], restored["bad_epochs"]
        best = read_checkpoint(output / "best.pt")
        best_metrics = best["validation_metrics"]
        best_score, best_loss = best_metrics["joint"]["macro_f1"], best_metrics["loss"]
        torch.set_rng_state(restored["torch_rng"])
        generator.set_state(restored["loader_rng"])
        random.setstate(restored["python_rng"])
        if device.type == "cuda" and restored["cuda_rng"]:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        if restored.get("finished"):
            plot_history(output / "history.png", history)
            plot_confusion(output / "validation_confusion_matrix.png", best_metrics, "Best validation checkpoint: fruit + condition")
            print("[DONE] This run already finished. See training_summary.json.", flush=True)
            return json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        print(f"[RESUME] Continuing after epoch {start_epoch}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "config.json", {**config, "dataset_identity": identity})
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "torchvision": str(torchvision.__version__), "cuda_runtime": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    atomic_json(output / "environment.json", environment)
    total_epochs = args.warmup_epochs + args.finetune_epochs
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, total_epochs):
            stage = "warmup" if epoch < args.warmup_epochs else "finetune"
            if stage != active_stage:
                optimizer = optimizer_for(model, stage)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-6) if stage == "finetune" else None
                active_stage, bad_epochs = stage, 0
            print(f"[EPOCH {epoch + 1}/{total_epochs}] {stage} | device={device} | batch={args.batch_size} x accumulation={args.accumulation}", flush=True)
            epoch_started = time.perf_counter()
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, stage, args.accumulation, args.label_smoothing)
            metrics, probabilities = evaluate_model(model, val_loader, device, args.label_smoothing)
            score = metrics["joint"]["macro_f1"]
            improved = score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and metrics["loss"] < best_loss - 1e-6)
            row = {"epoch": epoch + 1, "stage": stage, "train_loss": train_loss, "val_loss": metrics["loss"],
                   "val_joint_accuracy": metrics["joint"]["accuracy"], "val_fruit_accuracy": metrics["fruit"]["accuracy"],
                   "val_condition_accuracy": metrics["condition"]["accuracy"], "val_joint_macro_f1": score,
                   "seconds": time.perf_counter() - epoch_started, "new_best": improved}
            history.append(row)
            common = {**checkpoint_metadata(), "model_state": model.state_dict(), "dataset_identity": identity,
                      "config": config, "epoch": epoch + 1, "stage": stage, "validation_metrics": metrics,
                      "environment": environment, "pretrained_initialization": not args.from_scratch}
            if improved:
                best_score, best_loss, best_metrics, bad_epochs = score, metrics["loss"], metrics, 0
                save_checkpoint(output / "best.pt", common)
                atomic_json(output / "best_validation.json", {"epoch": epoch + 1, "stage": stage,
                    "split": "val", "model_selection": "joint_macro_f1_then_lower_validation_loss",
                    "dataset_identity": identity, "metrics": metrics})
                save_predictions(output, "validation", val_rows, probabilities)
            elif stage == "finetune":
                bad_epochs += 1
            if scheduler:
                scheduler.step()
            finished = epoch + 1 == total_epochs or (stage == "finetune" and bad_epochs >= args.patience)
            write_csv(output / "history.csv", history, list(history[0]))
            summary = {"status": "completed" if finished else "running", "completed_epochs": epoch + 1,
                       "best_epoch": json.loads((output / "best_validation.json").read_text())["epoch"],
                       "best_validation": best_metrics, "test_evaluated": False,
                       "model_selection_split": "val", "dataset_identity": identity,
                       "run_seconds_this_session": time.perf_counter() - started,
                       "pretrained_initialization": not args.from_scratch,
                       "best_model_sha256": file_sha(output / "best.pt"),
                       "stop_reason": "early_stopping" if finished and epoch + 1 < total_epochs else "epoch_limit" if finished else None}
            atomic_json(output / "training_summary.json", summary)
            save_checkpoint(output / "last.pt", {**common, "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None, "scaler_state": scaler.state_dict(),
                "history": history, "bad_epochs": bad_epochs, "finished": finished,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "loader_rng": generator.get_state(), "python_rng": random.getstate()})
            print(f"[VAL] fruit_acc={metrics['fruit']['accuracy']:.2%} | joint_acc={metrics['joint']['accuracy']:.2%} | joint_macro_f1={score:.4f}" + (" | BEST SAVED" if improved else ""), flush=True)
            if finished:
                break
    except KeyboardInterrupt:
        if (output / "last.pt").is_file():
            print("[PAUSED] Use the same command with --resume to continue from the last completed epoch.", flush=True)
        else:
            print("[PAUSED] No epoch has finished yet. Restart using a NEW --output directory.", flush=True)
        raise
    plot_history(output / "history.png", history)
    plot_confusion(output / "validation_confusion_matrix.png", best_metrics, "Best validation checkpoint: fruit + condition")
    print(f"[DONE] Best model: {output / 'best.pt'}", flush=True)
    print("[NEXT] Send training_summary.json and history.png. Test has not been evaluated.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="FreshLens EfficientNet-B0 fine-tuning")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "cnn_dataset_v3")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "models" / "cnn_efficientnet_b0")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-scratch", action="store_true", help="Testing/ablation only; the default uses pretrained weights")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (DataError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        if "out of memory" in str(exc).lower():
            print("Try a NEW output directory with --batch-size 4 --accumulation 4. Resume must use the original config.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
