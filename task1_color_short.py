import argparse
import os
from pathlib import Path

from dvm_color_classification import (
    CustomResNet18Like,
    build_class_weights,
    build_dataloaders,
    build_densenet121,
    build_mobilenet_v3_large,
    build_records,
    filter_classes,
    run_experiment,
    save_model_comparison,
    save_records_split,
    stratified_split,
    subset_records_stratified,
)
from util import apply_yaml_config, init_task_runtime, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: DVM car color classification")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--image_table_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="runs_task1_color")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu", "auto"])

    parser.add_argument("--front_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_samples_per_class", type=int, default=400)
    parser.add_argument("--max_classes", type=int, default=0)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--subset_size", type=int, default=0)
    parser.add_argument("--subset_min_per_class", type=int, default=4)

    parser.add_argument("--scratch_epochs", type=int, default=45)
    parser.add_argument("--finetune_epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_scratch", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_confusion", action="store_true")
    return parser.parse_args()


def main():
    args = apply_yaml_config(parse_args())
    if not args.data_root:
        raise RuntimeError("Set --data_root or provide it in --config")
    out_dir, device = init_task_runtime(args)
    os.environ.setdefault("TORCH_HOME", str(Path(out_dir) / ".torch_cache"))

    records = build_records(args.data_root, args.image_table_csv, args.front_only)
    records, class_to_idx, idx_to_class, _ = filter_classes(
        records, args.min_samples_per_class, args.max_classes
    )
    records = subset_records_stratified(
        records=records,
        subset_size=int(args.subset_size),
        seed=int(args.seed),
        min_per_class=int(args.subset_min_per_class),
    )
    if int(args.subset_size) > 0:
        present_classes = sorted({r.color for r in records})
        class_to_idx = {c: i for i, c in enumerate(present_classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

    train_records, val_records, test_records = stratified_split(
        records, args.test_size, args.val_size, args.seed
    )

    save_records_split(train_records, val_records, test_records, out_dir)
    print(f"Device: {device}")

    _, dataloaders = build_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        class_to_idx=class_to_idx,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_weights = build_class_weights(train_records, class_to_idx, device)

    experiments = [
        {
            "model_name": "custom_resnet18_like_scratch",
            "build_model_fn": lambda n: CustomResNet18Like(num_classes=n),
            "num_epochs": args.scratch_epochs,
            "lr": args.lr_scratch,
        },
        {
            "model_name": "mobilenet_v3_large_imagenet_finetune",
            "build_model_fn": build_mobilenet_v3_large,
            "num_epochs": args.finetune_epochs,
            "lr": args.lr_finetune,
        },
        {
            "model_name": "densenet121_imagenet_finetune",
            "build_model_fn": build_densenet121,
            "num_epochs": args.finetune_epochs,
            "lr": args.lr_finetune,
        },
    ]

    all_results = []
    for exp in experiments:
        print("\n" + "#" * 100)
        print(f"Running: {exp['model_name']}")
        result = run_experiment(
            model_name=exp["model_name"],
            build_model_fn=exp["build_model_fn"],
            dataloaders=dataloaders,
            class_weights=class_weights,
            idx_to_class=idx_to_class,
            device=device,
            num_epochs=exp["num_epochs"],
            lr=exp["lr"],
            weight_decay=args.weight_decay,
            patience=args.patience,
            out_dir=out_dir,
            save_confusion=args.save_confusion,
        )
        all_results.append(result)

    all_results = sorted(all_results, key=lambda x: x["test_f1_macro"], reverse=True)
    save_model_comparison(all_results, out_dir)
    save_json({"results": all_results}, Path(out_dir) / "task1_summary.json")

    print("\nBest model:")
    print(all_results[0])
    print(f"F1_macro target > 0.8: {all_results[0]['test_f1_macro'] > 0.8}")


if __name__ == "__main__":
    main()
