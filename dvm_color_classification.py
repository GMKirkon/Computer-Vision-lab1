import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import models
from torchvision import transforms as T

from util import IMAGE_EXTS, list_files


ImageFile.LOAD_TRUNCATED_IMAGES = True
FRONT_VIEWPOINT_ID = 0


@dataclass
class DVMRecord:
    img_path: Path
    color: str
    viewpoint: int | None


def parse_metadata_from_filename(path: Path) -> tuple[str, int | None] | None:
    parts = path.stem.split("$$")
    if len(parts) < 7:
        return None

    color = parts[3].strip().lower()
    if not color:
        return None

    viewpoint = None
    raw_view = parts[6].strip()
    if raw_view:
        try:
            viewpoint = int(raw_view)
        except ValueError:
            viewpoint = None

    return color, viewpoint


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _records_from_csv(
    image_paths: list[Path], image_table_csv: str | Path, front_only: bool
) -> list[DVMRecord]:
    by_name = {p.name: p for p in image_paths}
    records: list[DVMRecord] = []

    with Path(image_table_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col_map = {_normalize_col(c): c for c in cols}

        image_col = next(
            (
                col_map[k]
                for k in ("image_name", "image", "img_name")
                if k in col_map
            ),
            None,
        )
        color_col = next((col_map[k] for k in ("predicted_color", "color") if k in col_map), None)
        view_col = next(
            (
                col_map[k]
                for k in ("predicted_viewpoint", "viewpoint", "predicted_view")
                if k in col_map
            ),
            None,
        )

        if image_col is None:
            raise RuntimeError(f"Could not detect image column in {image_table_csv}")

        for row in reader:
            img_name = (row.get(image_col) or "").strip()
            if not img_name:
                continue

            path = by_name.get(img_name) or by_name.get(f"{img_name}.jpg")
            if path is None:
                continue

            color = (row.get(color_col) or "").strip().lower() if color_col else ""
            if not color:
                parsed = parse_metadata_from_filename(Path(img_name))
                color = parsed[0] if parsed else ""
            if not color:
                continue

            viewpoint = None
            if view_col:
                raw = (row.get(view_col) or "").strip()
                if raw:
                    try:
                        viewpoint = int(raw)
                    except ValueError:
                        viewpoint = None

            if front_only and viewpoint is not None and viewpoint != FRONT_VIEWPOINT_ID:
                continue

            records.append(DVMRecord(path, color, viewpoint))

    return records


def _records_from_filenames(image_paths: list[Path], front_only: bool) -> list[DVMRecord]:
    records: list[DVMRecord] = []
    for path in image_paths:
        parsed = parse_metadata_from_filename(path)
        if parsed is None:
            continue
        color, viewpoint = parsed
        if front_only and viewpoint is not None and viewpoint != FRONT_VIEWPOINT_ID:
            continue
        records.append(DVMRecord(path, color, viewpoint))
    return records


def build_records(data_root: str | Path, image_table_csv: str = "", front_only: bool = True) -> list[DVMRecord]:
    image_paths = list_files(data_root, IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError(f"No images found in {data_root}")

    if image_table_csv and Path(image_table_csv).exists():
        records = _records_from_csv(image_paths, image_table_csv, front_only)
    else:
        records = _records_from_filenames(image_paths, front_only)

    if not records:
        raise RuntimeError("No records left after metadata parsing/filtering")
    return records


def filter_classes(
    records: list[DVMRecord], min_samples_per_class: int, max_classes: int
) -> tuple[list[DVMRecord], dict[str, int], dict[int, str], Counter]:
    counts = Counter(r.color for r in records)

    valid = [c for c, n in counts.items() if n >= min_samples_per_class]
    valid = sorted(valid, key=lambda c: (-counts[c], c))
    if max_classes > 0:
        valid = valid[:max_classes]

    if len(valid) < 2:
        raise RuntimeError("Need at least 2 classes after filtering")

    filtered = [r for r in records if r.color in valid]
    class_to_idx = {c: i for i, c in enumerate(sorted(valid))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return filtered, class_to_idx, idx_to_class, counts


def subset_records_stratified(
    records: list[DVMRecord],
    subset_size: int,
    seed: int,
    min_per_class: int = 4,
) -> list[DVMRecord]:
    if subset_size <= 0 or subset_size >= len(records):
        return records
    if min_per_class < 1:
        min_per_class = 1

    full_counts = Counter(r.color for r in records)
    keep_classes = sorted(full_counts.keys(), key=lambda c: (-full_counts[c], c))

    while True:
        pool = [r for r in records if r.color in keep_classes]
        if len(pool) < subset_size:
            subset = pool
        else:
            labels = [r.color for r in pool]
            subset, _ = train_test_split(
                pool,
                train_size=subset_size,
                random_state=seed,
                stratify=labels,
            )

        subset_counts = Counter(r.color for r in subset)
        too_small = [c for c, n in subset_counts.items() if n < min_per_class]

        if len(subset_counts) >= 2 and len(too_small) == 0:
            return subset

        if len(keep_classes) <= 2:
            top2 = keep_classes[:2]
            pool2 = [r for r in records if r.color in top2]
            labels2 = [r.color for r in pool2]
            target = min(subset_size, len(pool2))
            subset2, _ = train_test_split(
                pool2,
                train_size=target,
                random_state=seed,
                stratify=labels2,
            )
            return subset2

        drop_candidates = too_small if too_small else keep_classes
        drop_class = min(drop_candidates, key=lambda c: full_counts[c])
        keep_classes = [c for c in keep_classes if c != drop_class]


def stratified_split(
    records: list[DVMRecord], test_size: float, val_size: float, seed: int
) -> tuple[list[DVMRecord], list[DVMRecord], list[DVMRecord]]:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1.0:
        raise ValueError("Expected split sizes >0 and test_size + val_size < 1")

    labels = [r.color for r in records]
    train_val, test = train_test_split(records, test_size=test_size, random_state=seed, stratify=labels)

    val_rel = val_size / (1.0 - test_size)
    train_labels = [r.color for r in train_val]
    train, val = train_test_split(train_val, test_size=val_rel, random_state=seed, stratify=train_labels)
    return train, val, test


class DVMColorDataset(Dataset):
    def __init__(self, records: list[DVMRecord], class_to_idx: dict[str, int], transform):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        with Image.open(rec.img_path) as img:
            x = self.transform(img.convert("RGB"))
        y = self.class_to_idx[rec.color]
        return x, y


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class CustomResNet18Like(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._stage(64, 64, blocks=2, stride=1)
        self.layer2 = self._stage(64, 128, blocks=2, stride=2)
        self.layer3 = self._stage(128, 256, blocks=2, stride=2)
        self.layer4 = self._stage(256, 512, blocks=2, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes))

    @staticmethod
    def _stage(in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        layers.extend(ResidualBlock(out_ch, out_ch, stride=1) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def build_mobilenet_v3_large(num_classes: int):
    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
    except AttributeError:
        weights = None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_densenet121(num_classes: int):
    try:
        weights = models.DenseNet121_Weights.DEFAULT
    except AttributeError:
        weights = None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


TASK1_MODEL_BUILDERS = {
    "custom_resnet18_like_scratch": lambda n: CustomResNet18Like(num_classes=n),
    "mobilenet_v3_large_imagenet_finetune": build_mobilenet_v3_large,
    "densenet121_imagenet_finetune": build_densenet121,
}


def class_names_from_split_csv(split_csv: str | Path, split: str = "test") -> list[str]:
    colors = set()
    with Path(split_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = str(row.get("split", "")).strip()
            if split and row_split != split:
                continue
            color = str(row.get("color", "")).strip().lower()
            if color:
                colors.add(color)
    names = sorted(colors)
    if not names:
        raise RuntimeError(f"No class names found in {split_csv} for split='{split}'")
    return names


def build_task1_eval_transform(img_size: int):
    return T.Compose(
        [
            T.Resize((int(img_size), int(img_size))),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_task1_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> nn.Module:
    if model_name not in TASK1_MODEL_BUILDERS:
        raise RuntimeError(f"Unsupported model name: {model_name}")

    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    model = TASK1_MODEL_BUILDERS[model_name](int(num_classes)).to(device_obj)

    checkpoint = torch.load(Path(checkpoint_path), map_location=device_obj)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict_task1_models_on_paths(
    checkpoints: dict[str, str | Path],
    class_names: list[str],
    image_paths: list[str | Path],
    img_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, list[str]]:
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    eval_tf = build_task1_eval_transform(int(img_size))
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    predictions: dict[str, list[str]] = {}
    for model_name, checkpoint_path in checkpoints.items():
        model = load_task1_model_from_checkpoint(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            num_classes=len(class_names),
            device=device_obj,
        )
        cur_preds = []
        for raw_path in image_paths:
            path = Path(raw_path)
            with Image.open(path) as im:
                x = eval_tf(im.convert("RGB")).unsqueeze(0).to(device_obj)
            with torch.inference_mode():
                idx = int(torch.argmax(model(x), dim=1).item())
            cur_preds.append(idx_to_class.get(idx, f"class_{idx}"))
        predictions[model_name] = cur_preds

    return predictions


def build_dataloaders(
    train_records: list[DVMRecord],
    val_records: list[DVMRecord],
    test_records: list[DVMRecord],
    class_to_idx: dict[str, int],
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            normalize,
        ]
    )
    eval_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), normalize])

    datasets = {
        "train": DVMColorDataset(train_records, class_to_idx, train_tf),
        "val": DVMColorDataset(val_records, class_to_idx, eval_tf),
        "test": DVMColorDataset(test_records, class_to_idx, eval_tf),
    }
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        ),
    }
    return datasets, dataloaders


def build_class_weights(train_records: list[DVMRecord], class_to_idx: dict[str, int], device: torch.device):
    counts = Counter(r.color for r in train_records)
    total = len(train_records)
    n_cls = len(class_to_idx)
    weights = [total / (n_cls * counts[c]) for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: optim.Optimizer | None = None,
    phase: str = "train",
    model_name: str = "",
    epoch: int | None = None,
    num_epochs: int | None = None,
):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    n_items = 0
    total_correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    if epoch is not None and num_epochs is not None:
        desc = f"[{model_name}] {phase} {epoch:02d}/{num_epochs}"
    else:
        desc = f"[{model_name}] {phase}"

    pbar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        pred = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        n_items += x.size(0)
        total_correct += int((pred == y).sum().item())
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        pbar.set_postfix(
            loss=f"{total_loss / max(1, n_items):.4f}",
            acc=f"{total_correct / max(1, n_items):.4f}",
        )

    conf = metrics.confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "loss": float(total_loss / max(1, n_items)),
        "acc": float(total_correct / max(1, n_items)),
        "f05_macro": float(metrics.fbeta_score(y_true, y_pred, average="macro", beta=0.5, zero_division=0)),
        "f1_macro": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "conf_matrix": conf.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _train(
    model: nn.Module,
    model_name: str,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    num_classes: int,
    num_epochs: int,
    patience: int,
    run_dir: Path,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"best_{model_name}.pth"

    history = {"train": [], "val": []}
    best_val_f1 = -1.0
    best_epoch = -1
    stale = 0
    print(
        f"[{model_name}] start training: epochs={num_epochs}, "
        f"patience={patience}, device={device.type}"
    )

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr = _run_epoch(
            model,
            dataloaders["train"],
            criterion,
            device,
            num_classes,
            optimizer=optimizer,
            phase="train",
            model_name=model_name,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        with torch.inference_mode():
            va = _run_epoch(
                model,
                dataloaders["val"],
                criterion,
                device,
                num_classes,
                phase="val",
                model_name=model_name,
                epoch=epoch,
                num_epochs=num_epochs,
            )

        history["train"].append({k: v for k, v in tr.items() if k not in {"y_true", "y_pred", "conf_matrix"}})
        history["val"].append({k: v for k, v in va.items() if k not in {"y_true", "y_pred", "conf_matrix"}})

        if scheduler is not None:
            scheduler.step()

        improved = va["f1_macro"] > best_val_f1
        if va["f1_macro"] > best_val_f1:
            best_val_f1 = va["f1_macro"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
        else:
            stale += 1

        dt = time.time() - t0
        mark = "*" if improved else ""
        print(
            f"[{model_name}] epoch {epoch:02d}/{num_epochs} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} f1 {tr['f1_macro']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1_macro']:.4f} | "
            f"best {best_val_f1:.4f}{mark} | stale {stale}/{patience} | {dt:.1f}s"
        )
        if stale >= patience:
            print(f"[{model_name}] early stop at epoch {epoch}")
            break

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[{model_name}] done: best_val_f1={best_val_f1:.4f} at epoch {best_epoch}")
    return ckpt_path, best_val_f1, best_epoch


def _evaluate(
    model: nn.Module,
    ckpt_path: Path,
    model_name: str,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_class: dict[int, str],
    num_classes: int,
    run_dir: Path,
    save_confusion: bool,
):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.inference_mode():
        ts = _run_epoch(
            model,
            dataloader,
            criterion,
            device,
            num_classes,
            phase="test",
            model_name=model_name,
        )

    report = metrics.classification_report(
        ts["y_true"],
        ts["y_pred"],
        labels=list(range(num_classes)),
        target_names=[idx_to_class[i] for i in range(num_classes)],
        digits=4,
        zero_division=0,
    )
    (run_dir / "test_report.txt").write_text(report, encoding="utf-8")

    if save_confusion:
        with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(ts["conf_matrix"])

    print(
        f"[{model_name}] test | loss {ts['loss']:.4f} "
        f"acc {ts['acc']:.4f} f1 {ts['f1_macro']:.4f}"
    )
    return {
        "test_loss": float(ts["loss"]),
        "test_acc": float(ts["acc"]),
        "test_f1_macro": float(ts["f1_macro"]),
        "test_f05_macro": float(ts["f05_macro"]),
        "best_val_f1": float(checkpoint.get("best_val_f1", -1.0)),
        "best_epoch": int(checkpoint.get("epoch", -1)),
    }


def run_experiment(
    model_name,
    build_model_fn,
    dataloaders,
    class_weights,
    idx_to_class,
    device,
    num_epochs,
    lr,
    weight_decay,
    patience,
    out_dir,
    save_confusion,
):
    num_classes = len(idx_to_class)
    model = build_model_fn(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    run_dir = Path(out_dir) / model_name
    ckpt_path, best_val_f1, best_epoch = _train(
        model=model,
        model_name=model_name,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        num_epochs=num_epochs,
        patience=patience,
        run_dir=run_dir,
    )

    test_metrics = _evaluate(
        model=model,
        ckpt_path=ckpt_path,
        model_name=model_name,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        idx_to_class=idx_to_class,
        num_classes=num_classes,
        run_dir=run_dir,
        save_confusion=save_confusion,
    )

    result = {"model_name": model_name, "best_val_f1": best_val_f1, "best_epoch": best_epoch}
    result.update(test_metrics)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def save_records_split(
    train_records: list[DVMRecord], val_records: list[DVMRecord], test_records: list[DVMRecord], out_dir: str | Path
):
    out_path = Path(out_dir) / "dataset_split.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "img_path", "color", "viewpoint"])
        for split, records in (("train", train_records), ("val", val_records), ("test", test_records)):
            for r in records:
                w.writerow([split, str(r.img_path), r.color, r.viewpoint])


def save_model_comparison(results: list[dict], out_dir: str | Path):
    out_dir = Path(out_dir)
    with (out_dir / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with (out_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_name",
                "best_val_f1",
                "best_epoch",
                "test_acc",
                "test_f1_macro",
                "test_f05_macro",
                "test_loss",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["model_name"],
                    r["best_val_f1"],
                    r["best_epoch"],
                    r["test_acc"],
                    r["test_f1_macro"],
                    r["test_f05_macro"],
                    r["test_loss"],
                ]
            )
