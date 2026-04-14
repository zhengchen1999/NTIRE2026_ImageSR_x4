#!/usr/bin/env python
from pathlib import Path
import argparse

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "mochunnian/SR"
REPO_ROOT = Path(__file__).resolve().parents[3] / "model_zoo" / "team16_APRIL-AIGC"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download submission assets from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo containing checkpoints/data.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--repo-type", default="model", help="Hugging Face repo type.")
    parser.add_argument(
        "--local-dir",
        default=str(REPO_ROOT),
        help="Directory where checkpoint/data assets should be materialized.",
    )
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        token=args.token,
        resume_download=True,
        allow_patterns=[
            "checkpoints/*",
            "checkpoints/**",
        ],
    )


if __name__ == "__main__":
    main()
