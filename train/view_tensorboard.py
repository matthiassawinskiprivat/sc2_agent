"""Launch TensorBoard for unit-control + build agent logs (used by view_tensorboard.bat)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Repository root (defaults to parent of train/)",
    )
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--print-spec-only",
        action="store_true",
        help="Print logdir_spec and exit (no TensorBoard).",
    )
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from train.async_train_universal import (
        build_viewer_tensorboard_logdir_spec,
        free_port,
    )

    logdir_spec = build_viewer_tensorboard_logdir_spec(repo_root)
    if args.print_spec_only:
        print(logdir_spec)
        return 0

    print(f"Freeing port {args.port} if needed...")
    free_port(args.port)

    print("TensorBoard logdir_spec:")
    print(f"  {logdir_spec}")
    print()
    print(f"Open: http://localhost:{args.port}")
    print("Sidebar runs:")
    print("  control / control_active  = unit-control (logs_universal)")
    print("  build / build_active      = build agent (logs_build_zerg)")
    print("Curves appear after the first learner update.")
    print()

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir_spec",
        logdir_spec,
        "--port",
        str(args.port),
        "--reload_interval",
        "15",
        "--load_fast",
        "false",
        "--bind_all",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
