from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import MultiInputImageDataset
from .metrics import LPIPSMetric, PSNRMetric, SSIM
from .model import build_model
from .utils import forward_with_tiling, parse_hw, save_image_tensor, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the multi-input image refiner.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--input-col", type=str, default="input_image")
    parser.add_argument("--target-col", type=str, default="gt")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--input-separator", type=str, default=",")
    parser.add_argument("--max-inputs", type=int, default=None)
    parser.add_argument("--resize", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--lpips-net", type=str, default="vgg")
    parser.add_argument("--flat-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-input-filename", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", "fusion_restormer")
    model_kwargs = checkpoint.get("model_kwargs", {})

    device = get_device(args.device)
    model = build_model(model_name, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    resize = parse_hw(args.resize)
    dataset = MultiInputImageDataset(
        csv_path=args.input_csv,
        root_dir=args.root_dir,
        input_col=args.input_col,
        target_col=args.target_col,
        id_col=args.id_col,
        input_separator=args.input_separator,
        max_inputs=args.max_inputs,
        patch_size=None,
        resize=resize,
        training=False,
        augment=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    if args.tile_size > 0 and args.batch_size != 1:
        raise ValueError("tile_size > 0 requires --batch-size 1")

    metrics_writer = None
    metrics_file = open(output_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=["sample_id", "psnr", "ssim", "lpips"])
    metrics_writer.writeheader()

    psnr_metric = PSNRMetric().to(device)
    ssim_metric = SSIM().to(device)
    lpips_metric = LPIPSMetric(args.lpips_net).to(device)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    counts = {"psnr": 0, "ssim": 0, "lpips": 0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="infer"):
            inputs = batch["inputs"].to(device, non_blocking=True)
            input_mask = batch["input_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            has_target = batch["has_target"].to(device)
            sample_ids = batch["sample_id"]
            input_paths = batch["input_paths"]

            prediction = forward_with_tiling(model, inputs, input_mask, tile_size=args.tile_size, tile_overlap=args.tile_overlap)

            for sample_index, sample_id in enumerate(sample_ids):
                if args.flat_output:
                    if args.use_input_filename:
                        primary_input_path = str(input_paths[sample_index]).split("|")[0]
                        save_path = output_dir / Path(primary_input_path).name
                    else:
                        save_path = output_dir / f"{sample_id}.png"
                else:
                    save_path = output_dir / "predictions" / f"{sample_id}.png"
                save_image_tensor(prediction[sample_index], save_path)

                row = {"sample_id": sample_id, "psnr": "", "ssim": "", "lpips": ""}
                if bool(has_target[sample_index].item()):
                    pred_sample = prediction[sample_index : sample_index + 1]
                    target_sample = target[sample_index : sample_index + 1]

                    psnr_value = float(psnr_metric(pred_sample, target_sample).item())
                    ssim_value = float(ssim_metric(pred_sample, target_sample).item())
                    row["psnr"] = psnr_value
                    row["ssim"] = ssim_value
                    totals["psnr"] += psnr_value
                    totals["ssim"] += ssim_value
                    counts["psnr"] += 1
                    counts["ssim"] += 1

                    if lpips_metric.available:
                        lpips_value = float(lpips_metric(pred_sample, target_sample).item())
                        row["lpips"] = lpips_value
                        totals["lpips"] += lpips_value
                        counts["lpips"] += 1

                metrics_writer.writerow(row)

    metrics_file.close()

    summary = {
        "psnr": totals["psnr"] / counts["psnr"] if counts["psnr"] > 0 else math.nan,
        "ssim": totals["ssim"] / counts["ssim"] if counts["ssim"] > 0 else math.nan,
        "lpips": totals["lpips"] / counts["lpips"] if counts["lpips"] > 0 else math.nan,
        "lpips_metric_name": f"pyiqa:{lpips_metric.metric_name if lpips_metric.available else args.lpips_net}",
        "num_samples": len(dataset),
    }
    save_json(output_dir / "metrics_summary.json", summary)


if __name__ == "__main__":
    main()
