from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    task1_dir = Path(__file__).resolve().parent
    config_path = task1_dir / "configs" / "task1_color.yaml"
    script_path = repo_root / "task1_color_short.py"

    if not config_path.exists():
        raise RuntimeError(f"Config not found: {config_path}")
    if not script_path.exists():
        raise RuntimeError(f"Training script not found: {script_path}")

    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str((repo_root / "runs_task1_color" / ".torch_cache").resolve()))

    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()
