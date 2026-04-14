"""Image generation inference script (supports TI2I with image conditioning).
- TI2I (text + single image conditioning for restoration-like tasks)
- Benchmark version: keep input/output filenames the same
- Support skip-if-exists + post-process color fix + multiple zips
"""

import argparse
import time
from pathlib import Path
import zipfile

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

from fastgen.configs.config import BaseConfig
from fastgen.third_party.color_refiner.model import build_model
from fastgen.third_party.color_refiner.utils import forward_with_tiling, save_image_tensor
import fastgen.utils.logging_utils as logger
from fastgen.utils import basic_utils
from fastgen.utils.distributed import clean_up
from fastgen.utils.scripts import parse_args, setup
from scripts.inference.inference_utils import (
    load_prompts,
    init_model,
    init_checkpointer,
    load_checkpoint,
    cleanup_unused_modules,
    setup_inference_modules,
    add_common_args,
)

import torchvision.transforms.functional as TF


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def to_rgb_if_rgba(img: Image.Image) -> Image.Image:
    if img.mode.upper() == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        return rgb_img
    return img


restoration_transform = transforms.Compose([
    transforms.Lambda(to_rgb_if_rgba),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] → [-1,1]
])


DEFAULT_PROMPTS = {
    "denoising" : "Remove strong Gaussian noise from the input image while preserving the exact scene content, natural colors, clean edges, fine textures, and realistic details. High-fidelity denoising, no plastic smoothing, no ringing, photorealistic restoration.",
    "sr" : "Restore a clean 4x super-resolution image from the low-resolution input while preserving the exact scene layout, natural colors, sharp edges, fine textures, and realistic details. High-fidelity upscaling, photorealistic restoration, no halos, no oversharpening.",
}


def collect_image_paths(image_dir: Path | None) -> list[Path]:
    if image_dir is None:
        return []
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def build_image_lookup(image_paths: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in image_paths:
        by_name[path.name] = path
        by_stem.setdefault(path.stem, []).append(path)
    return by_name, by_stem


def resolve_matching_image_path(
    reference_path: Path,
    by_name: dict[str, Path],
    by_stem: dict[str, list[Path]],
) -> Path | None:
    matched = by_name.get(reference_path.name)
    if matched is not None:
        return matched

    stem_matches = by_stem.get(reference_path.stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def maybe_upscale_input_image(image: Image.Image, task: str, input_upscale: int) -> Image.Image:
    if task != "sr" or input_upscale <= 1:
        return image
    return image.resize((image.width * input_upscale, image.height * input_upscale), Image.BICUBIC)


def save_passthrough_image(
    source_path: Path,
    save_path: Path,
    target_hw: tuple[int, int] | None = None,
) -> None:
    image = load_rgb_image(source_path)
    if target_hw is not None and (image.height, image.width) != target_hw:
        image = image.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def _prepare_condition(args, prompt, model, ctx, image_condition=None):
    text_prompts = [prompt] if prompt else [""]
    text_embeds = None
    if hasattr(model.net, "text_encoder"):
        with basic_utils.inference_mode(
            model.net.text_encoder,
            precision_amp=model.precision_amp_enc,
            device_type=model.device.type
        ):
            text_embeds = basic_utils.to(
                model.net.text_encoder.encode(text_prompts),
                **ctx
            )

    if image_condition is None:
        return text_embeds

    image_latents, image_latent_ids = model.net.prepare_img_conditioning(image_condition)

    return {
        "text_embeds": text_embeds,
        "image_latents": image_latents,
        "image_latent_ids": image_latent_ids,
    }


def load_refiner(checkpoint_path: str | None, device: torch.device) -> tuple[torch.nn.Module | None, dict | None]:
    if not checkpoint_path:
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "xxx")
    model_kwargs = checkpoint.get("model_kwargs", {})

    refiner = build_model(model_name, **model_kwargs).to(device)
    refiner.load_state_dict(checkpoint["model"], strict=True)
    refiner.eval()
    for parameter in refiner.parameters():
        parameter.requires_grad_(False)

    return refiner, {"model_name": model_name, "model_kwargs": model_kwargs}


def decode_latent_to_unit_interval(
    latent: torch.Tensor,
    vae,
    precision_amp: torch.dtype | None,
) -> torch.Tensor:
    with basic_utils.inference_mode(vae, precision_amp=precision_amp, device_type=latent.device.type):
        decoded = vae.decode(latent)
    return ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def refine_teacher_image(
    refiner: torch.nn.Module,
    teacher_image: torch.Tensor,
    tile_size: int,
    tile_overlap: int,
    image_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher_image = teacher_image.float().clamp(0.0, 1.0)

    if image_condition is None:
        refiner_inputs = teacher_image.unsqueeze(1)
    else:
        input_image = ((image_condition.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        refiner_inputs = torch.stack([input_image, teacher_image], dim=1)

    input_mask = torch.ones(
        (refiner_inputs.shape[0], refiner_inputs.shape[1]),
        device=refiner_inputs.device,
        dtype=refiner_inputs.dtype,
    )

    with torch.inference_mode():
        prediction = forward_with_tiling(
            refiner,
            refiner_inputs,
            input_mask,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    return prediction.clamp(0.0, 1.0)


def refine_prediction(
    refiner: torch.nn.Module,
    image_condition: torch.Tensor,
    generated_latent: torch.Tensor,
    vae,
    precision_amp: torch.dtype | None,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
    teacher_image = decode_latent_to_unit_interval(generated_latent, vae=vae, precision_amp=precision_amp)
    return refine_teacher_image(
        refiner=refiner,
        teacher_image=teacher_image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        image_condition=image_condition,
    )


def should_use_tiled_teacher_sampling(
    tile_size_latent: int,
    height: int,
    width: int,
    direct_max_size: int,
) -> bool:
    if tile_size_latent <= 0:
        return False
    if direct_max_size > 0 and height * width <= direct_max_size * direct_max_size:
        return False
    return True


def save_final_prediction(
    save_path: Path,
    generated_latent: torch.Tensor,
    image_condition: torch.Tensor,
    vae,
    precision_amp: torch.dtype | None,
    refiner: torch.nn.Module | None,
    tile_size: int,
    tile_overlap: int,
    need_resize: bool,
    orig_h: int,
    orig_w: int,
) -> None:
    if refiner is None:
        basic_utils.save_media(generated_latent, str(save_path), vae=vae, precision_amp=precision_amp)
        if need_resize:
            restored_img = Image.open(save_path).convert("RGB")
            restored_img = TF.resize(restored_img, size=(orig_h, orig_w))
            restored_img.save(save_path)
        return
    # import pdb; pdb.set_trace()
    prediction = refine_prediction(
        refiner=refiner,
        image_condition=image_condition,
        generated_latent=generated_latent,
        vae=vae,
        precision_amp=precision_amp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if need_resize:
        prediction = F.interpolate(prediction, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    save_image_tensor(prediction[0], save_path)


def main(args, config: BaseConfig):
    if not args.input_image_dir and not args.diffusion_input_dir:
        raise ValueError("Either --input_image_dir or --diffusion_input_dir is required")

    input_dir = Path(args.input_image_dir) if args.input_image_dir else None
    diffusion_input_dir = Path(args.diffusion_input_dir) if args.diffusion_input_dir else None

    input_paths = collect_image_paths(input_dir) if input_dir is not None else []
    diffusion_input_paths = collect_image_paths(diffusion_input_dir) if diffusion_input_dir is not None else []
    use_precomputed_diffusion = diffusion_input_dir is not None

    if args.test_vae and use_precomputed_diffusion:
        raise ValueError("--test_vae is not supported with --diffusion_input_dir")
    if not input_paths and not diffusion_input_paths:
        raise ValueError("No images found for inference")

    prompt = DEFAULT_PROMPTS.get(args.task, "A high-quality restored image.")
    primary_paths = input_paths if input_paths else diffusion_input_paths
    pos_prompt_set = [prompt] * len(primary_paths)
    logger.info(f"TI2I mode: {len(primary_paths)} images, task={args.task}, prompt={prompt[:60]}...")

    diffusion_by_name: dict[str, Path] = {}
    diffusion_by_stem: dict[str, list[Path]] = {}
    if use_precomputed_diffusion:
        diffusion_by_name, diffusion_by_stem = build_image_lookup(diffusion_input_paths)
        logger.info(f"Using precomputed diffusion images from {diffusion_input_dir}")
        if not input_paths:
            if args.refiner_ckpt:
                logger.info("No original input images provided. Refiner will run with diffusion images only.")
            else:
                logger.info("No refiner checkpoint provided. Diffusion images will be exported directly.")

    save_dir = Path(args.image_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")
    original_zip = save_dir.parent / f"{save_dir.name}.zip"

    # 如果你希望「连原始 zip 都存在就不跑推理」，可以在这里加：
    # if original_zip.exists():
    #     logger.info(f"Original zip already exists: {original_zip} → skip inference")
    #     # 但我们这里选择继续跑（以支持只补 color fix 的场景）
    #     # 如果你想要严格跳过，可以取消注释上面三行

    model = None
    teacher = None
    student = None
    vae = None
    ctx: dict[str, torch.dtype | torch.device] | None = None
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_size_latent = 0
    tile_overlap_latent = 0

    if use_precomputed_diffusion:
        refiner, refiner_meta = load_refiner(args.refiner_ckpt, runtime_device)
        has_teacher = False
        has_student = False
    else:
        basic_utils.set_random_seed(config.trainer.seed, by_rank=True)
        if args.guidance_scale is not None:
            config.model.guidance_scale = args.guidance_scale

        model = init_model(config)
        checkpointer = init_checkpointer(config)
        load_checkpoint(checkpointer, model, args.ckpt_path, config)

        cleanup_unused_modules(model, args.do_teacher_sampling)

        teacher, student, vae = setup_inference_modules(
            model, config, args.do_teacher_sampling, args.do_student_sampling, model.precision
        )
        ctx = {"dtype": model.precision, "device": model.device}
        runtime_device = model.device
        refiner, refiner_meta = load_refiner(args.refiner_ckpt, model.device)

        tile_size_latent = 0 if args.tile_size <= 0 else max(1, (args.tile_size + 15) // 16)
        tile_overlap_latent = 0 if args.tile_overlap <= 0 else max(0, args.tile_overlap // 16)

        if tile_size_latent > 0:
            logger.info(
                f"Tiled teacher sampling enabled: tile_size={args.tile_size}px ({tile_size_latent} latent), "
                f"tile_overlap={args.tile_overlap}px ({tile_overlap_latent} latent)"
            )
        else:
            logger.info("Tiled teacher sampling disabled")
        if args.teacher_direct_max_size > 0:
            logger.info(
                "Teacher tiled sampling will be skipped when inference size is within "
                f"{args.teacher_direct_max_size}x{args.teacher_direct_max_size}"
            )

        has_teacher = teacher is not None and (hasattr(teacher, "sample") or hasattr(teacher, "sample_tiled"))
        has_student = student is not None and hasattr(model, "generator_fn")
        assert has_teacher or has_student, "Need at least one sampler"

    if refiner is not None:
        logger.info(
            f"Loaded refiner: {refiner_meta['model_name']} from {args.refiner_ckpt} "
            f"with tile_size={args.refiner_tile_size}, tile_overlap={args.refiner_tile_overlap}"
        )
    runtimes: list[float] = []

    # ────────────────────────────────────────────────
    # 主推理循环（带跳过已存在文件）
    # ────────────────────────────────────────────────
    for i, prompt in enumerate(pos_prompt_set):
        img_path = primary_paths[i]
        logger.info(f"[{i+1}/{len(primary_paths)}] Processing: {img_path.name}")

        save_path = save_dir / img_path.name

        if save_path.exists() and not args.overwrite:
            logger.info(f"Already exists, skipping inference: {save_path.name}")
            continue
        if save_path.exists() and args.overwrite:
            logger.info(f"Overwrite enabled, regenerating: {save_path.name}")

        if use_precomputed_diffusion:
            original_path = img_path if input_paths else None
            diffusion_path = img_path if not input_paths else resolve_matching_image_path(
                img_path,
                diffusion_by_name,
                diffusion_by_stem,
            )
            if diffusion_path is None:
                raise FileNotFoundError(
                    f"Could not find a matching diffusion image for {img_path.name} in {diffusion_input_dir}"
                )

            target_hw: tuple[int, int] | None = None
            image_condition: torch.Tensor | None = None
            diffusion_image = load_rgb_image(diffusion_path)

            if original_path is not None:
                original_image = maybe_upscale_input_image(
                    load_rgb_image(original_path),
                    task=args.task,
                    input_upscale=args.input_upscale,
                )
                target_hw = (original_image.height, original_image.width)
                if (diffusion_image.height, diffusion_image.width) != target_hw:
                    logger.info(
                        f"Resizing diffusion image from {diffusion_image.height}x{diffusion_image.width} "
                        f"to {target_hw[0]}x{target_hw[1]} for {img_path.name}"
                    )
                    diffusion_image = diffusion_image.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
                image_condition = restoration_transform(original_image).unsqueeze(0).to(device=runtime_device)
            else:
                target_hw = (diffusion_image.height, diffusion_image.width)

            start = time.time()
            if refiner is None:
                save_passthrough_image(diffusion_path, save_path, target_hw=target_hw)
            else:
                teacher_image = transforms.ToTensor()(diffusion_image).unsqueeze(0).to(device=runtime_device)
                prediction = refine_teacher_image(
                    refiner=refiner,
                    teacher_image=teacher_image,
                    tile_size=args.refiner_tile_size,
                    tile_overlap=args.refiner_tile_overlap,
                    image_condition=image_condition,
                )
                save_image_tensor(prediction[0], save_path)
            runtimes.append(time.time() - start)
            logger.info(f"Saved: {save_path}")
            continue

        assert model is not None and ctx is not None and vae is not None

        pil_img = maybe_upscale_input_image(
            load_rgb_image(img_path),
            task=args.task,
            input_upscale=args.input_upscale,
        )
        tensor_img = restoration_transform(pil_img)
        _, orig_H, orig_W = tensor_img.shape
        need_resize = (orig_H % 16 != 0) or (orig_W % 16 != 0)

        if need_resize:
            H = ((orig_H + 15) // 16) * 16
            W = ((orig_W + 15) // 16) * 16
            
            logger.info(f"Original size {orig_H}x{orig_W} not divisible by 16, "
                        f"resizing to {H}x{W} for inference")

            tensor_img = TF.resize(
                tensor_img,
                size=(H, W),
            )
        else:
            H, W = orig_H, orig_W

        image_condition = tensor_img.unsqueeze(0).unsqueeze(2)     # [1, C, 1, H, W]
        image_condition = basic_utils.to(image_condition, **ctx)

        condition = _prepare_condition(args, prompt, model, ctx, image_condition=image_condition)
        neg_condition = _prepare_condition(args, args.neg_prompt , model, ctx, image_condition=image_condition)

        if isinstance(condition, dict) and "image_latents" in condition:
            noise_shape = [1, 128, H // 16, W // 16]
        else:
            noise_shape = [1, *config.model.input_shape]

        noise = torch.randn(noise_shape, **ctx)

        if args.denoising_strength < 1.0:
            condition_image_latent = model.net.vae.encode(image_condition.squeeze(2))
            s = args.denoising_strength
            noisy_latent = (1 - s) ** 0.5 * condition_image_latent + s ** 0.5 * noise
        else:
            noisy_latent = noise

        
        if args.test_vae:
            condition_image_latent = model.net.vae.encode(image_condition.squeeze(2))
            basic_utils.save_media(condition_image_latent, str(save_path), vae=vae, precision_amp=model.precision_amp_infer)
            logger.info(f"Saved: {save_path}")
            if need_resize:
                restored_img = Image.open(save_path).convert("RGB")
                restored_img = TF.resize(
                    restored_img,
                    size=(orig_H, orig_W),
                )
                restored_img.save(save_path)  # png 建议不设 quality，jpg 才用
                logger.info(f"Resized back to original {orig_H}x{orig_W} and overwrote: {save_path}")
                
            continue


        # Teacher sampling (main branch)
        if has_teacher and not args.do_student_sampling:
            start = time.time()
            teacher_kwargs = {
                "num_steps": args.num_steps,
                "second_order": False,
                "precision_amp": model.precision_amp_infer,
            }
            if config.model.skip_layers:
                teacher_kwargs["skip_layers"] = config.model.skip_layers

            use_tiled_teacher_sampling = hasattr(teacher, "sample_tiled") and should_use_tiled_teacher_sampling(
                tile_size_latent=tile_size_latent,
                height=H,
                width=W,
                direct_max_size=args.teacher_direct_max_size,
            )
            if (
                tile_size_latent > 0
                and hasattr(teacher, "sample_tiled")
                and args.teacher_direct_max_size > 0
                and H * W <= args.teacher_direct_max_size * args.teacher_direct_max_size
            ):
                logger.info(
                    f"Skip tiled teacher sampling for {img_path.name}: "
                    f"inference size {H}x{W} <= {args.teacher_direct_max_size}x{args.teacher_direct_max_size}"
                )

            if use_tiled_teacher_sampling:
                with basic_utils.inference_mode(
                    teacher,
                    precision_amp=model.precision_amp_infer,
                    device_type=model.device.type,
                ):
                    img_teacher = teacher.sample_tiled(
                        noisy_latent,
                        condition=condition,
                        neg_condition=neg_condition,
                        guidance_scale=config.model.guidance_scale,
                        tile_size=tile_size_latent,
                        tile_overlap=tile_overlap_latent,
                        **teacher_kwargs,
                    ).to(dtype=noisy_latent.dtype)
            else:
                img_teacher = model.sample(
                    teacher,
                    noisy_latent,
                    condition=condition,
                    neg_condition=neg_condition,
                    **teacher_kwargs,
                )
            logger.info(f"Teacher sampling time: {time.time() - start:.2f}s")
            save_final_prediction(
                save_path=save_path,
                generated_latent=img_teacher,
                image_condition=image_condition.squeeze(2),
                vae=vae,
                precision_amp=model.precision_amp_infer,
                refiner=refiner,
                tile_size=args.refiner_tile_size,
                tile_overlap=args.refiner_tile_overlap,
                need_resize=need_resize,
                orig_h=orig_H,
                orig_w=orig_W,
            )
            runtimes.append(time.time() - start)
            logger.info(f"Saved: {save_path}")
                

        # Student sampling (可选，视 --do_student_sampling)
        if has_student and args.do_student_sampling:

            start_time = time.time()
            image_student = model.generator_fn(
                student,
                noisy_latent,
                condition=condition,
                student_sample_steps=model.config.student_sample_steps if args.num_student_steps is None else args.num_student_steps,
                student_sample_type=model.config.student_sample_type,
                t_list=model.config.sample_t_cfg.t_list,
                precision_amp=model.precision_amp_infer,
            )
            logger.info(f"Student sampling time: {time.time() - start_time:.2f}s")
            save_final_prediction(
                save_path=save_path,
                generated_latent=image_student,
                image_condition=image_condition.squeeze(2),
                vae=vae,
                precision_amp=model.precision_amp_infer,
                refiner=refiner,
                tile_size=args.refiner_tile_size,
                tile_overlap=args.refiner_tile_overlap,
                need_resize=need_resize,
                orig_h=orig_H,
                orig_w=orig_W,
            )
            runtimes.append(time.time() - start_time)
            logger.info(f"Saved: {save_path}")
                

    # ────────────────────────────────────────────────
    # Color Fix 后处理 + 各自的 zip
    # ────────────────────────────────────────────────

    # source_dir = Path(args.input_image_dir)

    # for fix_name, fix_func in [
    #     ("wavelet", wavelet_color_fix),
    #     ("adain", adain_color_fix),
    # ]:
    #     fix_dir = save_dir.parent / f"{save_dir.name}_{fix_name}_color_fix"
    #     fix_dir.mkdir(parents=True, exist_ok=True)
    #     logger.info(f"Color fix ({fix_name}) → {fix_dir}")

    #     for img_path in input_paths:
    #         target_path = save_dir / img_path.name
    #         if not target_path.exists():
    #             logger.warning(f"Generated image not found, skip color fix: {img_path.name}")
    #             continue

    #         fix_path = fix_dir / img_path.name
    #         if fix_path.exists():
    #             logger.info(f"Color fixed file already exists, skip: {fix_path.name}")
    #             continue

    #         try:
    #             source_img = Image.open(img_path).convert("RGB")
    #             target_img = Image.open(target_path).convert("RGB")
    #             fixed_img = fix_func(target_img, source_img)
    #             fixed_img.save(fix_path, quality=95)
    #             logger.info(f"Color fixed ({fix_name}): {fix_path.name}")
    #         except Exception as e:
    #             logger.error(f"Color fix ({fix_name}) failed for {img_path.name}: {e}")

    #     # 打包该 color fix 目录
    #     fix_zip = fix_dir.parent / f"{fix_dir.name}.zip"
    #     if fix_zip.exists():
    #         logger.info(f"{fix_name} zip already exists: {fix_zip}")
    #     else:
    #         with zipfile.ZipFile(fix_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    #             for f in fix_dir.glob("*.[jpJP][pnPN][gG]"):
    #                 zf.write(f, arcname=f.name)
    #         logger.info(f"Created zip: {fix_zip}")

    # ────────────────────────────────────────────────
    # 最后打包原始结果（如果还没打包）
    # ────────────────────────────────────────────────

    readme_path = save_dir / "readme.txt"

    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0

    readme_content = (
        f"runtime per image [s] : {avg_runtime:.2f}\n"
        f"CPU[1] / GPU[0] : 0\n"
        f"Extra Data [1] / No Extra Data [0] : 0\n"
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.rstrip("\n"))   # 去掉最後多餘的換行（可選）

    logger.info(f"Generated readme.txt → {readme_path}")

    zip_path = Path(args.zip_path) if args.zip_path else original_zip

    if zip_path.exists():
        logger.info(f"Original zip already exists: {zip_path}")
    else:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in save_dir.glob("*"):
                zf.write(f, arcname=f.name)
        logger.info(f"Created original submission zip: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with TI2I support + color fix + incremental run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument("--denoising_strength", default=1.0, type=float)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--image_save_dir", default=None, type=str, required=True)
    parser.add_argument("--num_steps", default=28, type=int, help="teacher sampling steps")
    parser.add_argument("--num_student_steps", default=None, type=int, help="Student sampling steps")
    parser.add_argument("--guidance_scale", default=None, type=float, help="teacher guidance_scale")
    parser.add_argument("--refiner_ckpt", default=None, type=str, help="Color refiner checkpoint path")
    parser.add_argument("--refiner_tile_size", default=0, type=int, help="Tile size for refiner inference")
    parser.add_argument("--refiner_tile_overlap", default=32, type=int, help="Tile overlap for refiner inference")
    parser.add_argument(
        "--diffusion_input_dir",
        default=None,
        type=str,
        help="Optional directory with precomputed diffusion outputs. When set, diffusion sampling is skipped.",
    )
    parser.add_argument("--zip_path", default=None, type=str, help="Optional zip output path")
    parser.add_argument(
        "--tile_size",
        default=0,
        type=int,
        help="Tiled teacher sampling size in image-space pixels. Set 0 to disable.",
    )
    parser.add_argument(
        "--tile_overlap",
        default=64,
        type=int,
        help="Tiled teacher sampling overlap in image-space pixels.",
    )
    parser.add_argument(
        "--teacher_direct_max_size",
        default=0,
        type=int,
        help="Skip tiled teacher sampling when both inference height and width are <= this value. Set 0 to disable.",
    )

    parser.add_argument("--ti2i", action="store_true")
    parser.add_argument("--input_image_dir", type=str, default=None)
    parser.add_argument(
        "--input_upscale",
        default=1,
        type=int,
        help="Upscale the input image by this factor before inference. Set 4 for SR x4 bicubic upsampling.",
    )
    parser.add_argument("--task", type=str, default="ai_flash_portrait",
                        choices=list(DEFAULT_PROMPTS.keys()))
    parser.add_argument("--test_vae", action="store_true")
    parser.add_argument(
        "--neg_prompt",
        default="",
        type=str,
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)


    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)
    clean_up()
