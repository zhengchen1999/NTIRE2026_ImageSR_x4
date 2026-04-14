from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .dataset import MultiInputImageDataset
from .losses import RefinerLoss
from .metrics import LPIPSMetric, PSNRMetric, SSIM
from .model import build_model
from .utils import append_jsonl, forward_with_tiling, parse_hw, save_image_strip, save_image_tensor, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-input image refiner.")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, default="")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--input-col", type=str, default="input_image")
    parser.add_argument("--target-col", type=str, default="gt")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--input-separator", type=str, default=",")
    parser.add_argument("--max-inputs", type=int, default=None)
    parser.add_argument("--patch-size", type=int, nargs="+", default=[256])
    parser.add_argument("--resize", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-all-epochs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-val-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-image-every", type=int, default=1)
    parser.add_argument("--val-image-limit", type=int, default=8)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-model-only", type=str, default="")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--model", type=str, default="restormer")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--stage-dims", type=int, nargs=4, default=None)
    parser.add_argument("--num-blocks", type=int, nargs=4, default=[4, 6, 6, 8])
    parser.add_argument("--num-heads", type=int, nargs=4, default=[1, 2, 4, 8])
    parser.add_argument("--ffn-expansion", type=float, nargs="+", default=[2.66])
    parser.add_argument("--refinement-blocks", type=int, default=4)
    parser.add_argument("--base-input-index", type=int, default=-1)
    parser.add_argument("--use-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--ssim-loss-weight", type=float, default=0.2)
    parser.add_argument("--color-loss-weight", type=float, default=0.1)
    parser.add_argument("--lpips-loss-weight", type=float, default=0.0)
    parser.add_argument("--lpips-net", type=str, default="alex")
    return parser.parse_args()


def create_dataloader(
    dataset: MultiInputImageDataset,
    batch_size: int,
    num_workers: int,
    training: bool,
    distributed: bool = False,
) -> DataLoader:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=num_workers > 0,
    )


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def load_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    if isinstance(model, (nn.DataParallel, DDP)):
        expects_prefixed = True
    else:
        expects_prefixed = False

    has_prefixed = bool(state_dict) and next(iter(state_dict)).startswith("module.")
    if expects_prefixed and not has_prefixed:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    elif not expects_prefixed and has_prefixed:
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


def build_model_only_checkpoint(
    model: nn.Module,
    *,
    epoch: int,
    best_psnr: float,
    model_name: str,
    model_kwargs: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "best_psnr": best_psnr,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "args": vars(args),
    }


def save_checkpoint_bundle(
    *,
    save_dir: str | Path,
    checkpoint: dict[str, object],
    model_only_checkpoint: dict[str, object],
    epoch: int,
    save_all_epochs: bool,
) -> None:
    save_dir_path = Path(save_dir)
    torch.save(checkpoint, save_dir_path / "last.pt")
    torch.save(model_only_checkpoint, save_dir_path / "model_only.pt")
    if save_all_epochs:
        epoch_name = f"epoch_{epoch + 1:04d}"
        torch.save(checkpoint, save_dir_path / f"{epoch_name}.pt")
        torch.save(model_only_checkpoint, save_dir_path / f"{epoch_name}_model_only.pt")


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = bool(args.ddp or world_size > 1)

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = get_device(args.device)
        world_size = dist.get_world_size()
        return True, local_rank, world_size, device

    return False, local_rank, 1, get_device(args.device)


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_epoch_stats(
    running: dict[str, float],
    sample_count: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    if not (dist.is_available() and dist.is_initialized()):
        return running, sample_count

    stats = torch.tensor(
        [
            running["loss"],
            running["pixel"],
            running["ssim"],
            running["color"],
            running["lpips"],
            float(sample_count),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (
        {
            "loss": float(stats[0].item()),
            "pixel": float(stats[1].item()),
            "ssim": float(stats[2].item()),
            "color": float(stats[3].item()),
            "lpips": float(stats[4].item()),
        },
        int(stats[5].item()),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr: float, base_lr: float):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + cosine * (1.0 - min_lr / base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_validation(
    model: torch.nn.Module,
    criterion: RefinerLoss,
    dataloader: DataLoader,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    psnr_metric: PSNRMetric,
    ssim_metric: SSIM,
    lpips_metric: LPIPSMetric | None,
    save_dir: Path | None = None,
    epoch: int | None = None,
    save_image_limit: int = 0,
) -> dict[str, float]:
    model.eval()

    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    count = 0
    lpips_count = 0
    saved_images = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="validate", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            input_mask = batch["input_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            has_target = batch["has_target"].to(device)
            if not bool(has_target.all()):
                continue

            prediction = forward_with_tiling(model, inputs, input_mask, tile_size=tile_size, tile_overlap=tile_overlap)
            loss, _ = criterion(prediction, target)

            batch_size = inputs.shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            totals["psnr"] += float(psnr_metric(prediction, target).item()) * batch_size
            totals["ssim"] += float(ssim_metric(prediction, target).item()) * batch_size

            if lpips_metric is not None and lpips_metric.available:
                totals["lpips"] += float(lpips_metric(prediction, target).item()) * batch_size
                lpips_count += batch_size

            if save_dir is not None and epoch is not None and saved_images < save_image_limit:
                epoch_dir = save_dir / f"epoch_{epoch:04d}"
                sample_ids = batch["sample_id"]
                remaining = save_image_limit - saved_images
                for sample_index, sample_id in enumerate(sample_ids):
                    if remaining <= 0:
                        break
                    safe_sample_id = str(sample_id).replace("/", "_").replace("\\", "_")
                    valid_inputs = int(input_mask[sample_index].sum().item())
                    visual_tensors = [inputs[sample_index, input_index] for input_index in range(valid_inputs)]
                    visual_tensors.extend([prediction[sample_index], target[sample_index]])

                    for input_index in range(valid_inputs):
                        save_image_tensor(
                            inputs[sample_index, input_index],
                            epoch_dir / f"{safe_sample_id}_input{input_index}.png",
                        )
                    save_image_tensor(prediction[sample_index], epoch_dir / f"{safe_sample_id}_pred.png")
                    save_image_tensor(target[sample_index], epoch_dir / f"{safe_sample_id}_gt.png")
                    save_image_strip(visual_tensors, epoch_dir / f"{safe_sample_id}_compare.png")

                    saved_images += 1
                    remaining -= 1

            count += batch_size

    if count == 0:
        return {"loss": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}

    metrics = {
        "loss": totals["loss"] / count,
        "psnr": totals["psnr"] / count,
        "ssim": totals["ssim"] / count,
        "lpips": totals["lpips"] / lpips_count if lpips_count > 0 else float("nan"),
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.resume and args.resume_model_only:
        raise ValueError("Use either --resume or --resume-model-only, not both.")
    distributed, local_rank, world_size, device = setup_distributed(args)

    try:
        seed_everything(args.seed + (local_rank if distributed else 0))
        if is_main_process():
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        patch_size = parse_hw(args.patch_size)
        resize = parse_hw(args.resize)
        amp_enabled = bool(args.amp and device.type == "cuda")

        if args.tile_size > 0 and args.val_batch_size != 1:
            raise ValueError("tile_size > 0 requires --val-batch-size 1")

        train_dataset = MultiInputImageDataset(
            csv_path=args.train_csv,
            root_dir=args.root_dir,
            input_col=args.input_col,
            target_col=args.target_col,
            id_col=args.id_col,
            input_separator=args.input_separator,
            max_inputs=args.max_inputs,
            patch_size=patch_size,
            resize=resize,
            training=True,
            augment=args.augment,
        )
        train_loader = create_dataloader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            training=True,
            distributed=distributed,
        )

        val_loader = None
        if args.val_csv and is_main_process():
            val_dataset = MultiInputImageDataset(
                csv_path=args.val_csv,
                root_dir=args.root_dir,
                input_col=args.input_col,
                target_col=args.target_col,
                id_col=args.id_col,
                input_separator=args.input_separator,
                max_inputs=args.max_inputs or train_dataset.max_inputs,
                patch_size=None,
                resize=resize,
                training=False,
                augment=False,
            )
            val_loader = create_dataloader(val_dataset, args.val_batch_size, args.num_workers, training=False)

        expansion_factor: float | tuple[float, ...]
        if len(args.ffn_expansion) == 1:
            expansion_factor = args.ffn_expansion[0]
        else:
            expansion_factor = tuple(args.ffn_expansion)

        model_kwargs = {
            "embed_dim": args.embed_dim,
            "stage_dims": tuple(args.stage_dims) if args.stage_dims is not None else None,
            "num_blocks": tuple(args.num_blocks),
            "num_heads": tuple(args.num_heads),
            "expansion_factor": expansion_factor,
            "refinement_blocks": args.refinement_blocks,
            "base_input_index": args.base_input_index,
            "use_residual": args.use_residual,
            "gradient_checkpointing": args.gradient_checkpointing,
        }
        model = build_model(args.model, **model_kwargs).to(device)
        if distributed:
            if device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters,
                )
            else:
                model = DDP(model, find_unused_parameters=args.find_unused_parameters)
        elif args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        criterion = RefinerLoss(
            pixel_weight=args.pixel_loss_weight,
            ssim_weight=args.ssim_loss_weight,
            color_weight=args.color_loss_weight,
            lpips_weight=args.lpips_loss_weight,
            lpips_net=args.lpips_net,
        ).to(device)
        psnr_metric = PSNRMetric().to(device)
        ssim_metric = SSIM().to(device)
        lpips_metric = LPIPSMetric(args.lpips_net).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
        scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr, args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        start_epoch = 0
        best_psnr = float("-inf")

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            load_model_state(model, checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        elif args.resume_model_only:
            checkpoint = torch.load(args.resume_model_only, map_location="cpu")
            load_model_state(model, checkpoint["model"])

        if is_main_process():
            config_payload = vars(args).copy()
            config_payload["patch_size"] = list(patch_size) if patch_size is not None else None
            config_payload["resize"] = list(resize) if resize is not None else None
            config_payload["train_dataset_max_inputs"] = train_dataset.max_inputs
            config_payload["world_size"] = world_size
            config_payload["effective_batch_size"] = int(args.batch_size) * max(1, int(args.grad_accum_steps)) * (
                world_size if distributed else 1
            )
            save_json(Path(args.save_dir) / "config.json", config_payload)

        history_path = Path(args.save_dir) / "history.jsonl"
        checkpoint = None
        model_only_checkpoint = None

        for epoch in range(start_epoch, args.epochs):
            model.train()
            if distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            running = {"loss": 0.0, "pixel": 0.0, "ssim": 0.0, "color": 0.0, "lpips": 0.0}
            sample_count = 0
            progress = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}", disable=not is_main_process())
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(progress, start=1):
                inputs = batch["inputs"].to(device, non_blocking=True)
                input_mask = batch["input_mask"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    prediction = model(inputs, input_mask)
                    loss, components = criterion(prediction, target)

                scaled_loss = loss / max(1, args.grad_accum_steps)
                scaler.scale(scaled_loss).backward()

                should_step = (step % max(1, args.grad_accum_steps) == 0) or (step == len(train_loader))
                if should_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                batch_size = inputs.shape[0]
                running["loss"] += float(loss.item()) * batch_size
                running["pixel"] += float(components.get("pixel", torch.tensor(0.0)).item()) * batch_size
                running["ssim"] += float(components.get("ssim", torch.tensor(0.0)).item()) * batch_size
                running["color"] += float(components.get("color", torch.tensor(0.0)).item()) * batch_size
                running["lpips"] += float(components.get("lpips", torch.tensor(0.0)).item()) * batch_size
                sample_count += batch_size

                if is_main_process() and step % args.log_interval == 0:
                    progress.set_postfix(
                        loss=f"{running['loss'] / sample_count:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

            scheduler.step()
            running, sample_count = reduce_epoch_stats(running, sample_count, device)

            if is_main_process():
                train_metrics = {
                    "epoch": epoch + 1,
                    "split": "train",
                    "loss": running["loss"] / sample_count,
                    "pixel": running["pixel"] / sample_count,
                    "ssim_loss": running["ssim"] / sample_count,
                    "color_loss": running["color"] / sample_count,
                    "lpips_loss": running["lpips"] / sample_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                append_jsonl(history_path, train_metrics)

                checkpoint = {
                    "epoch": epoch,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_psnr": best_psnr,
                    "model_name": args.model,
                    "model_kwargs": model_kwargs,
                    "args": vars(args),
                }
                model_only_checkpoint = build_model_only_checkpoint(
                    model,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    model_name=args.model,
                    model_kwargs=model_kwargs,
                    args=args,
                )

                if val_loader is not None and (epoch + 1) % args.eval_every == 0:
                    val_image_dir = None
                    if args.save_val_images and (epoch + 1) % args.val_image_every == 0 and args.val_image_limit > 0:
                        val_image_dir = Path(args.save_dir) / "val_images"
                    val_metrics = run_validation(
                        model=model,
                        criterion=criterion,
                        dataloader=val_loader,
                        device=device,
                        tile_size=args.tile_size,
                        tile_overlap=args.tile_overlap,
                        psnr_metric=psnr_metric,
                        ssim_metric=ssim_metric,
                        lpips_metric=lpips_metric if lpips_metric.available else None,
                        save_dir=val_image_dir,
                        epoch=epoch + 1,
                        save_image_limit=args.val_image_limit,
                    )
                    val_record = {"epoch": epoch + 1, "split": "val", **val_metrics}
                    append_jsonl(history_path, val_record)

                    if val_metrics["psnr"] > best_psnr:
                        best_psnr = val_metrics["psnr"]
                        checkpoint["best_psnr"] = best_psnr
                        model_only_checkpoint["best_psnr"] = best_psnr
                        torch.save(checkpoint, Path(args.save_dir) / "best_psnr.pt")
                        torch.save(model_only_checkpoint, Path(args.save_dir) / "best_psnr_model_only.pt")

                if (epoch + 1) % args.save_every == 0:
                    checkpoint["best_psnr"] = best_psnr
                    model_only_checkpoint["best_psnr"] = best_psnr
                    save_checkpoint_bundle(
                        save_dir=args.save_dir,
                        checkpoint=checkpoint,
                        model_only_checkpoint=model_only_checkpoint,
                        epoch=epoch,
                        save_all_epochs=args.save_all_epochs,
                    )

            barrier_if_needed()

        if is_main_process() and checkpoint is not None and model_only_checkpoint is not None:
            checkpoint["best_psnr"] = best_psnr
            model_only_checkpoint["best_psnr"] = best_psnr
            save_checkpoint_bundle(
                save_dir=args.save_dir,
                checkpoint=checkpoint,
                model_only_checkpoint=model_only_checkpoint,
                epoch=int(checkpoint["epoch"]),
                save_all_epochs=False,
            )
    finally:
        barrier_if_needed()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
