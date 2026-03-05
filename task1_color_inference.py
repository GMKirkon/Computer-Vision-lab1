import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from dvm_color_classification import (
    CustomResNet18Like,
    build_densenet121,
    build_mobilenet_v3_large,
)
from util import IMAGE_EXTS, list_files, save_json, select_torch_device


MODEL_BUILDERS = {
    "custom_resnet18_like_scratch": lambda n: CustomResNet18Like(num_classes=n),
    "mobilenet_v3_large_imagenet_finetune": build_mobilenet_v3_large,
    "densenet121_imagenet_finetune": build_densenet121,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 inference: car color classification")
    parser.add_argument("--weights", type=str, required=True, help="Path to best_*.pth checkpoint")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory")
    parser.add_argument("--out_dir", type=str, default="runs_task1_color/inference")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu", "auto"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--model_name",
        type=str,
        default="auto",
        choices=[
            "auto",
            "custom_resnet18_like_scratch",
            "mobilenet_v3_large_imagenet_finetune",
            "densenet121_imagenet_finetune",
        ],
    )
    parser.add_argument(
        "--dataset_split_csv",
        type=str,
        default="",
        help="Optional dataset_split.csv from training run",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default="",
        help="Optional comma-separated class names. Used if dataset_split.csv is unavailable.",
    )
    return parser.parse_args()


def infer_model_name(weights_path: Path, checkpoint: dict, model_name_arg: str) -> str:
    if model_name_arg != "auto":
        return model_name_arg
    ckpt_name = str(checkpoint.get("model_name", "")).strip()
    if ckpt_name in MODEL_BUILDERS:
        return ckpt_name

    stem = weights_path.stem.lower()
    for name in MODEL_BUILDERS:
        if name in stem:
            return name
    raise RuntimeError("Cannot infer model_name. Pass --model_name explicitly.")


def infer_dataset_split_csv(weights_path: Path) -> Path | None:
    # Expected layout: runs_task1_color/<model_name>/best_<model_name>.pth
    candidate = weights_path.parent.parent / "dataset_split.csv"
    if candidate.exists():
        return candidate
    return None


def class_names_from_dataset_split(csv_path: Path) -> list[str]:
    colors = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            color = str(row.get("color", "")).strip().lower()
            if color:
                colors.add(color)
    names = sorted(colors)
    if not names:
        raise RuntimeError(f"No classes found in {csv_path}")
    return names


def load_class_names(args, weights_path: Path) -> list[str]:
    if args.class_names.strip():
        out = [x.strip().lower() for x in args.class_names.split(",") if x.strip()]
        if len(out) < 2:
            raise RuntimeError("Need at least 2 class names in --class_names.")
        return sorted(set(out))

    if args.dataset_split_csv:
        csv_path = Path(args.dataset_split_csv)
    else:
        csv_path = infer_dataset_split_csv(weights_path)

    if csv_path is None or not csv_path.exists():
        raise RuntimeError(
            "dataset_split.csv not found. Pass --dataset_split_csv or --class_names explicitly."
        )
    return class_names_from_dataset_split(csv_path)


def collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f"Unsupported file extension: {p.suffix}")
        return [p]
    if p.is_dir():
        images = list_files(p, IMAGE_EXTS)
        if not images:
            raise RuntimeError(f"No images found in {p}")
        return images
    raise RuntimeError(f"Source path not found: {source}")


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise RuntimeError(f"Weights not found: {weights_path}")
    if not weights_path.is_file():
        raise RuntimeError(
            f"Weights path must be a checkpoint file (*.pth), got: {weights_path}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_torch_device(args.device)
    checkpoint = torch.load(weights_path, map_location=device)

    class_names = load_class_names(args, weights_path)
    num_classes = len(class_names)
    model_name = infer_model_name(weights_path, checkpoint, args.model_name)
    model = MODEL_BUILDERS[model_name](num_classes).to(device)

    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    eval_tf = T.Compose(
        [
            T.Resize((int(args.img_size), int(args.img_size))),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    images = collect_images(args.source)
    topk = max(1, min(int(args.topk), num_classes))
    rows = []
    top1_counter = Counter()

    for img_path in images:
        with Image.open(img_path) as im:
            x = eval_tf(im.convert("RGB")).unsqueeze(0).to(device)

        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1)[0].detach().cpu()
            vals, idxs = torch.topk(probs, k=topk)

        topk_preds = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            topk_preds.append({"class_id": int(idx), "class_name": class_names[int(idx)], "prob": float(score)})

        top1 = topk_preds[0]
        top1_counter[top1["class_name"]] += 1
        rows.append({"image": str(img_path), "top1": top1, "topk": topk_preds})
        print(f"{img_path.name}\t{top1['class_name']}\t{top1['prob']:.4f}")

    summary = {
        "weights": str(weights_path),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "images_total": len(rows),
        "top1_class_histogram": dict(top1_counter),
        "predictions": rows,
    }
    save_json(summary, out_dir / "task1_inference_summary.json")
    print(f"\nSaved summary: {(out_dir / 'task1_inference_summary.json').resolve()}")


if __name__ == "__main__":
    main()
