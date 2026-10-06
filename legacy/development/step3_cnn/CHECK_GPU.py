"""Check a real forward/backward pass, without downloading pretrained weights."""

import argparse
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check PyTorch and GPU support")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent.parent / "reports" / "cnn_environment.json")
    args = parser.parse_args()
    try:
        import torch
        import torchvision
        from cnn_data import atomic_json
        from cnn_model import choose_device, make_model

        device = choose_device(args.device)
        model = make_model(pretrained=False).to(device).train()
        images = torch.randn(2, 3, 224, 224, device=device)
        targets = torch.tensor([0, 1], device=device)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        result = {"python": platform.python_version(), "torch": str(torch.__version__),
                  "torchvision": str(torchvision.__version__), "cuda_available": torch.cuda.is_available(),
                  "cuda_runtime": torch.version.cuda, "requested_device": args.device,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "gpu_memory_gib": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2) if device.type == "cuda" else None,
                  "forward_backward_ok": bool(torch.isfinite(loss)), "output_shape": list(logits.shape)}
        atomic_json(args.report, result)
        if not result["forward_backward_ok"]:
            raise RuntimeError("Forward/backward check returned a non-finite loss.")
        print(f"[OK] PyTorch {torch.__version__}; torchvision {torchvision.__version__}")
        print(f"[OK] Device: {result['gpu'] or 'CPU'}; forward/backward passed")
        print(f"[OK] Report: {args.report}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        print("Install the packages using the commands in HUONG_DAN.md, then run this check again.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
