"""Export a FastGen checkpoint into a diffusers transformer directory."""

import argparse

from fastgen.configs.config import BaseConfig
from fastgen.utils import basic_utils
from fastgen.utils.distributed import clean_up
from fastgen.utils.scripts import parse_args, setup
from scripts.inference.inference_utils import (
    init_model,
    init_checkpointer,
    load_checkpoint,
    cleanup_unused_modules,
    setup_inference_modules,
    add_common_args,
)
import fastgen.utils.logging_utils as logger




def main(args, config: BaseConfig):
    basic_utils.set_random_seed(config.trainer.seed, by_rank=True)
    if args.guidance_scale is not None:
        config.model.guidance_scale = args.guidance_scale

    model = init_model(config)
    # model.guidance_scale
    checkpointer = init_checkpointer(config)

    load_checkpoint(checkpointer, model, args.ckpt_path, config)


    cleanup_unused_modules(model, args.do_teacher_sampling)

    teacher, student, vae = setup_inference_modules(
        model, config, args.do_teacher_sampling, args.do_student_sampling, model.precision
    )
    ctx = {"dtype": model.precision, "device": model.device}

    has_teacher = teacher is not None and hasattr(teacher, "sample")
    assert has_teacher, "Teacher sampler is required when exporting the transformer"

    core_model = teacher.transformer
    core_model.save_pretrained(
        args.diffuser_output_dir,
        safe_serialization=True,
    )
    logger.success(f"Saved diffusers transformer to {args.diffuser_output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with TI2I support + color fix + incremental run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument("--guidance_scale", default=None, type=float, help="teacher guidance_scale")
    parser.add_argument("--diffuser_output_dir", type=str, default=None)
    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)
    clean_up()
