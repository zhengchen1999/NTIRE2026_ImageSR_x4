from __future__ import annotations

import argparse

from . import hat_backend


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run team01_CIPLAB step1 (HAT-L) inference.")
    parser.add_argument("input_path", type=str, help="Image directory or dataset root containing an `LQ` folder.")
    parser.add_argument("output_path", type=str, help="Directory where HAT upscaled images will be written.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device string (e.g. cuda:0). Defaults to auto-detect.",
    )
    args = parser.parse_args(argv)

    hat_backend.run_from_input_dir(args.input_path, args.output_path, device=args.device)


if __name__ == "__main__":
    main()

