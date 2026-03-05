import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_yaml_config(args: Any) -> Any:
    cfg_path = getattr(args, "config", "")
    if not cfg_path:
        return args

    cfg = load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        return args

    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def list_files(root: str | Path, exts: set[str]) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def select_torch_device(device: str = "auto") -> torch.device:
    mps_available = (
        hasattr(torch.backends, "mps")
        and hasattr(torch.backends.mps, "is_available")
        and torch.backends.mps.is_available()
    )

    if device == "mps":
        return torch.device("mps") if mps_available else torch.device("cpu")
    if device == "cpu":
        return torch.device("cpu")

    # auto
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def init_task_runtime(args: Any) -> tuple[Path, torch.device]:
    seed_everything(int(args.seed))
    out_dir = ensure_dir(args.out_dir)
    device = select_torch_device(getattr(args, "device", "auto"))
    return out_dir, device
