import copy
from pathlib import Path

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import SRx4_ImageLoaderConfig
from fastgen.configs.net import Flux2_klein_base4BConfig


REPO_ROOT = Path(__file__).resolve().parents[6]

LOCAL_MODEL_DIR = REPO_ROOT / "model_zoo" / "team16_APRIL-AIGC" / "checkpoints" / "flux2_klein_base4b_srx4"


def create_config():
    config = config_sft_default.create_config()

    config.model.precision = "bfloat16"
    config.model.input_shape = [128, -1, -1]
    config.model.cond_dropout_prob = 0.15
    config.model.cond_keys_no_dropout = ["image_latents", "image_latent_ids"]

    config.model.net = copy.deepcopy(Flux2_klein_base4BConfig)
    config.model.net.model_id = str(LOCAL_MODEL_DIR)

    config.model.net_optimizer.lr = 1e-5
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.guidance_scale = 2.0
    config.model.student_sample_steps = 10

    config.dataloader_train = SRx4_ImageLoaderConfig
    config.dataloader_train.batch_size = 4
    config.dataloader_train.batch_size_config = {"768": 4, "768x768": 4}
    config.dataloader_train.resolutions = ["768"]
    config.dataloader_train.exact_resolutions = ["768x768"]

    config.trainer.fsdp = False
    config.trainer.ddp = False
    config.model.fsdp_meta_init = False
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 500
    config.trainer.save_ckpt_iter = 500
    config.trainer.max_keep_ckpts = 2

    config.log_config.project = "submit_exp"
    config.log_config.group = "submit"
    config.log_config.name = "sr_step_explore"

    return config
