from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Task1 inference launcher (best model auto-pick)")
    parser.add_argument("--source", type=str, default="data/dvm_fronts/confirmed_fronts")
    parser.add_argument("--out_dir", type=str, default="runs_task1_color/inference_best")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu", "auto"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--summary_path",
        type=str,
        default="runs_task1_color/task1_summary.json",
        help="Path to Task1 summary with model ranking",
    )
    return parser.parse_args()


def pick_best_weights(summary_path: Path, repo_root: Path) -> Path:
    if not summary_path.exists():
        raise RuntimeError(f"Summary not found: {summary_path}")

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No results in summary: {summary_path}")

    best = sorted(results, key=lambda x: float(x.get("test_f1_macro", 0.0)), reverse=True)[0]
    model_name = str(best.get("model_name", "")).strip()
    if not model_name:
        raise RuntimeError("Best model_name is missing in summary")

    weights = repo_root / "runs_task1_color" / model_name / f"best_{model_name}.pth"
    if not weights.exists():
        raise RuntimeError(f"Best weights not found: {weights}")
    return weights


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    infer_script = repo_root / "task1_color_inference.py"
    summary_path = (repo_root / args.summary_path).resolve()

    if not infer_script.exists():
        raise RuntimeError(f"Inference script not found: {infer_script}")

    weights = pick_best_weights(summary_path=summary_path, repo_root=repo_root)
    cmd = [
        sys.executable,
        str(infer_script),
        "--weights",
        str(weights),
        "--source",
        str((repo_root / args.source).resolve()),
        "--out_dir",
        str((repo_root / args.out_dir).resolve()),
        "--device",
        args.device,
        "--img_size",
        str(int(args.img_size)),
        "--topk",
        str(int(args.topk)),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
