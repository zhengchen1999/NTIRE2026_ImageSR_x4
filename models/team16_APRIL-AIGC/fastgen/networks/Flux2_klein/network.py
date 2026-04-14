# ssj180123@gmail.com 2026.02.10
# diffusers-0.37.0.dev0: https://github.com/huggingface/diffusers.git@64e2adf

import os
from typing import Any, Optional, List, Set, Union, Tuple, Dict
import types

import torch
import torch.utils.checkpoint
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLFlux2
from diffusers.models import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


class Flux2TextEncoder:
    """Text encoder for Flux2 using Qwen3 model."""

    def __init__(self, model_id: str):
        # Qwen3 text encoder
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="tokenizer",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="text_encoder",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder.eval().requires_grad_(False)


    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 512,
        hidden_states_layers: List[int] = (9, 18, 27),
    ) -> torch.Tensor:
        """Encode text prompts to embeddings.
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        all_input_ids = []
        all_attention_masks = []

        for prompt in conditioning:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.text_encoder.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.text_encoder.device)

        # 2. Forward pass
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            
        out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(precision)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds


    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.text_encoder.to(*args, **kwargs)
        return self


class Flux2ImageEncoder:
    """VAE encoder/decoder for Flux2.

    Flux VAE uses both scaling_factor and shift_factor for latent normalization.
    """

    def __init__(self, model_id: str):
        self.vae: AutoencoderKLFlux2 = AutoencoderKLFlux2.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="vae",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.vae.eval().requires_grad_(False)


    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents


    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents
        
    # flux 2的 vae有特殊的设计, _patchify_latents要先于normalization (不同于qwen img和flux1)
    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space + patchify.
        """
        image_latents = self.vae.encode(real_images, return_dict=False)[0].sample()
        image_latents = self._patchify_latents(image_latents)
        # Apply Flux2-specific mean and std
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """unpatchify + Decode latents to images.
        """
        # Reverse Flux-specific shift and scale
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        images = self.vae.decode(latents, return_dict=False)[0].clip(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.vae.to(*args, **kwargs)
        return self


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[dict] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Modified forward pass for Flux2Transformer2DModel with feature extraction support.
    Returns:
        Model output, optionally with features or logvar.
    """
    if feature_indices is None:
        feature_indices = set()

    if return_features_early and len(feature_indices) == 0:
        return []

    idx, features = 0, []

    # Store original sequence length to compute spatial dims for feature reshaping
    # hidden_states: [B, seq_len, C*4] where seq_len = (H//2) * (W//2)
    seq_len = hidden_states.shape[1]
    # if spatial_size is None:
    #     spatial_size = int(seq_len**0.5)  # Assuming square spatial dimensions

    # 1. Calculate timestep embedding and modulation parameters
    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)[0] 

    # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 3. Calculate RoPE embeddings from image and text tokens
    # NOTE: the below logic means that we can't support batched inference with images of different resolutions or
    # text prompts of differents lengths. Is this a use case we want to support?
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    num_txt_tokens = encoder_hidden_states.shape[1]


    # 4. Double Stream Transformer Blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Check if we should extract features at this index
        if idx in feature_indices:
            feat = hidden_states.clone()
            # B, S, C = feat.shape
            # feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)

        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1
    # Concat
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 5. Single Stream Transformer Blocks
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                single_stream_mod,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if idx in feature_indices:
            feat = hidden_states[:, num_txt_tokens:, :].clone()
            # B, S_img, C = feat.shape
            # feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)

        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1

    # Remove text tokens from concatenated stream
    hidden_states = hidden_states[:, num_txt_tokens:, ...]


    # 6. Output layers
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)


    if return_features_early:
        assert len(features) == len(feature_indices)
        return features

    if len(feature_indices) == 0:
        out = output
    else:
        out = [output, features] if features else output

    if return_logvar:
        logvar = self.logvar_linear(temb)
        return out, logvar

    return out

class Flux2_klein(FastGenNetwork):
    """Flux.2 network for text-to-image generation.

    Reference: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
    """

    MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        guidance_scale: Optional[float] = 2.0,
        load_pretrained: bool = True,
        # is_img2img: bool = True,
        **model_kwargs,
    ):
        """Flux.2
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self._disable_grad_ckpt = disable_grad_ckpt
        # self.is_img2img =is_img2img
        logger.debug(f"Embedded guidance scale: {guidance_scale}")

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id, load_pretrained)

        # Override forward with classify_forward
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        # Disable cuDNN SDPA backend to avoid mha_graph->execute errors during backward.
        # This is a known issue with Flux transformer and cuDNN attention.
        # Flash and mem_efficient backends still work; only cuDNN is problematic.
        if torch.backends.cuda.is_built():
            torch.backends.cuda.enable_cudnn_sdp(False)
            logger.info("Disabled cuDNN SDPA backend for Flux compatibility")

        # Gradient checkpointing configuration
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        # import pdb;pdb.set_trace()
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading Flux2 transformer from pretrained")
            self.transformer: Flux2Transformer2DModel = Flux2Transformer2DModel.from_pretrained(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            # If we're in a meta context, tensors will automatically be on meta device
            config = Flux2Transformer2DModel.load_config(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info(
                    "Initializing Flux transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Flux transformer from config (no pretrained weights)")
                logger.warning("Flux transformer being initialized from config. No weights are loaded!")
            self.transformer: Flux2Transformer2DModel = Flux2Transformer2DModel.from_config(config)

        # Add logvar linear layer for variance estimation - Flux uses 3072-dim time embeddings
        self.transformer.logvar_linear = torch.nn.Linear(3072, 1)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization.

        This is required when using meta device initialization for FSDP2.
        Reinitializes all linear layers and embeddings.
        """
        import torch.nn as nn

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()

        logger.debug("Reinitialized Flux parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the Flux network for FSDP.

        Note: Flux has two types of transformer blocks:
        - transformer_blocks: Joint attention blocks for text-image interaction
        - single_transformer_blocks: Single stream blocks for image processing

        We shard `self.transformer` instead of `self` because the network wrapper
        class may have complex multiple inheritance with ABC, which causes Python's
        __class__ assignment to fail due to incompatible memory layouts.
        """
        # Note: Checkpointing has to happen first, for proper casting during backward pass recomputation.
        if self.transformer.gradient_checkpointing:
            # Disable the built-in gradient checkpointing (which uses torch.utils.checkpoint)
            self.transformer.disable_gradient_checkpointing()
            # Apply FSDP-compatible activation checkpointing to both block types
            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda block: isinstance(block, (Flux2TransformerBlock, Flux2SingleTransformerBlock)),
            )
            logger.info("Applied FSDP activation checkpointing to Flux transformer blocks")

        # Apply FSDP sharding to joint transformer blocks
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)

        # Apply FSDP sharding to single transformer blocks
        for block in self.transformer.single_transformer_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Flux."""
        self.text_encoder = Flux2TextEncoder(model_id=self.model_id)

    def init_vae(self):
        """Initialize only the VAE for visualization."""
        self.vae = Flux2ImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def _prepare_latent_image_ids(
        self,
        latents: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids.to(device)


    def _prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D) or (L, D)
        device: torch.device,
        t_coord: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids).to(device)

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # b c h w -> b h*w c

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    def _unpack_latents_with_ids(self, x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)


    def prepare_img_conditioning(
        self,
        image_condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        for img2img conditioning latents。
        Args:
            image_condition: (B, C, T, H, W)
            
        Returns:
            image_latents:    (B, total_tokens, C)    pack + concat
            image_latent_ids: (1, total_tokens, 4)    ids
        """
        if image_condition.ndim != 5:
            raise ValueError(f"Expected image_condition shape (B, C, T, H, W), got {image_condition.shape}")

        B, C_img, T, H, W = image_condition.shape

        flat_images = image_condition.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C_img, H, W)

        latents = self.vae.encode(flat_images)  # (B*T, C_patched, h, w)

        B_T, C_patched, h, w = latents.shape
        latents = latents.view(B, T, C_patched, h, w)   # (B, T, C, h, w)

        packed_flat = self._pack_latents(latents.view(B * T, C_patched, h, w))   # (B*T, h*w, C_patched)
        packed = packed_flat.view(B, T, h * w, C_patched)                        # (B, T, patches, C)

        image_latents = packed.reshape(B, T * (h * w), C_patched)                # (B, total_tokens, C)

        image_latent_ids = self._prepare_reference_image_ids(
            num_refs=T,
            height=h,
            width=w,
        )
        image_latent_ids = image_latent_ids.unsqueeze(0).expand(B, -1, -1)
        image_latent_ids = image_latent_ids.to(latents.device)


        return image_latents, image_latent_ids

    def _prepare_reference_image_ids(
        self,
        num_refs: int,
        height: int,
        width: int,
        scale: int = 10,
    ) -> torch.Tensor:
 
        image_latent_ids = []

        for i in range(num_refs):

            t = torch.tensor([scale + scale * i])

            h = torch.arange(height)
            w = torch.arange(width)
            l = torch.arange(1)

            coords = torch.cartesian_prod(t, h, w, l)   # shape: (height*width, 4)

            image_latent_ids.append(coords)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)          # (total_patches, 4)
        return image_latent_ids
        
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Union[torch.Tensor, Dict[str, Any]] = None,
        r: Optional[torch.Tensor] = None,  # unused, kept for API compatibility
        guidance: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:


        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        batch_size = x_t.shape[0]
        height, width = x_t.shape[2], x_t.shape[3]

        # Handle dict-style condition for image2image mode
        image_latents = None
        image_latent_ids= None
        if isinstance(condition, dict):
            text_embeds = condition["text_embeds"]
            image_latents = condition["image_latents"]
            image_latent_ids = condition["image_latent_ids"]
        else:
            text_embeds = condition

        # Prepare position IDs
        latent_ids = self._prepare_latent_image_ids(x_t, x_t.device)
        txt_ids = self._prepare_text_ids(text_embeds, x_t.device)

        # Pack latents for transformer: [B, C, H, W] -> [B, H*W, C]
        hidden_states = self._pack_latents(x_t)
        main_seq_len = hidden_states.shape[1]


        if image_latents is not None:
            hidden_states = torch.cat([hidden_states, image_latents], dim=1)
            img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        else:
            img_ids = latent_ids


        # Note: Flux.2 guidance to model is None

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=text_embeds,
            timestep=t,  # Flux expects timestep in [0, 1]
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            # model_outputs [[B, seq_len, C],[B, seq_len, C],[B, seq_len, C]...]
            sliced_features = []
            for feat in model_outputs:
                # [B, H*W, C] -> [B, C, H, W]
                sliced_features.append(self._unpack_latents_with_ids(feat[:, :main_seq_len, :], latent_ids))
            return sliced_features

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        
        # Unpack output: [B, H*W, C] -> [B, C, H, W]
        if isinstance(out, torch.Tensor):
            out = out[:, :main_seq_len, :]
            out = self._unpack_latents_with_ids(out, latent_ids)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            out[0] = out[0][:, :main_seq_len, :]
            out[0] = self._unpack_latents_with_ids(out[0], latent_ids)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            processed_feats = []
            for feat in out[1]:
                sliced = feat[:, :main_seq_len, :]
                unpacked = self._unpack_latents_with_ids(sliced, latent_ids)
                processed_feats.append(unpacked)
            out = (out[0], processed_feats)

        if return_logvar:

            return out, logvar

        return out

    def compute_empirical_mu(
        self,
        image_seq_len: int,
        num_steps: int,
    ) -> float:
        """Copied from Flux.2 official compute_empirical_mu"""
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs, 
    ):
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed")

        if timesteps is not None:
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        elif sigmas is not None:
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps

        return timesteps, len(timesteps)

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 2.0,
        num_steps: int = 30,
        **kwargs,
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        """Generate samples using Euler flow matching.
        """
        batch_size, channels, height, width = noise.shape

        # Calculate image sequence length for shift calculation
        image_seq_len = height * width

        # Calculate resolution-dependent shift (mu)
        mu = self.compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)

        # Initialize scheduler with proper shift
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler" 
        )
        # scheduler.set_timesteps(num_steps, device=noise.device)
        # timesteps = scheduler.timesteps
        # 官方实现
        timesteps, num_inference_steps = self.retrieve_timesteps(
            scheduler,
            num_inference_steps=num_steps,
            device=noise.device,
            mu=mu,                     # ← 关键传入
        )

        # Initialize latents with proper scaling based on the initial timestep
        t_init = self.noise_scheduler.safe_clamp(
            timesteps[0] / 1000.0, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t
        )
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        # Sampling loop
        for timestep in timesteps:
            # Scheduler timesteps are in [0, 1000], transformer expects [0, 1]
            t = (timestep / 1000.0).expand(batch_size)
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            # Two guidance modes:
            # 1. CFG mode: when neg_condition is provided (uses uncond/cond difference)
            # 2. Guidance distillation mode: when neg_condition is None (single forward, guidance embedded)
            noise_pred = self(
                latents,
                t,
                condition,
                fwd_pred_type="flow",
                guidance=None,
            )

            if neg_condition is not None:
                # Traditional CFG mode
                noise_pred_uncond = self(
                    latents,
                    t,
                    neg_condition,
                    fwd_pred_type="flow",
                    guidance=None,
                )
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
         

            # Euler step
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents

    @torch.no_grad()
    def sample_tiled(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 2.0,
        num_steps: int = 30,
        tile_size: int = 0,
        tile_overlap: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching with latent-space tiling."""
        batch_size, channels, height, width = noise.shape

        if tile_size is None or tile_size <= 0 or (height <= tile_size and width <= tile_size):
            return self.sample(
                noise=noise,
                condition=condition,
                neg_condition=neg_condition,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                **kwargs,
            )

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        tile_overlap = 0 if tile_overlap is None else int(tile_overlap)
        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ValueError(f"`tile_size` must be positive, got {tile_size}")
        tile_overlap = max(0, min(tile_overlap, tile_size - 1))
        stride = tile_size - tile_overlap
        if stride <= 0:
            raise ValueError(
                f"`tile_overlap` must be smaller than `tile_size`, got tile_overlap={tile_overlap}, tile_size={tile_size}"
            )

        def get_tile_starts(length: int) -> List[int]:
            if length <= tile_size:
                return [0]
            starts = list(range(0, length - tile_size + 1, stride))
            last_start = length - tile_size
            if starts[-1] != last_start:
                starts.append(last_start)
            return starts

        def crop_condition(
            cond: Optional[Union[torch.Tensor, Dict[str, Any]]],
            top: int,
            left: int,
            bottom: int,
            right: int,
        ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
            if not isinstance(cond, dict):
                return cond

            image_latents = cond.get("image_latents")
            image_latent_ids = cond.get("image_latent_ids")
            if image_latents is None or image_latent_ids is None:
                return cond

            mask = (
                (image_latent_ids[0, :, 1] >= top)
                & (image_latent_ids[0, :, 1] < bottom)
                & (image_latent_ids[0, :, 2] >= left)
                & (image_latent_ids[0, :, 2] < right)
            )

            cropped = dict(cond)
            cropped["image_latents"] = image_latents[:, mask, :]
            cropped_ids = image_latent_ids[:, mask, :].clone()
            cropped_ids[:, :, 1] -= top
            cropped_ids[:, :, 2] -= left
            cropped["image_latent_ids"] = cropped_ids
            return cropped

        def get_tile_weight(top: int, left: int, bottom: int, right: int) -> torch.Tensor:
            tile_h = bottom - top
            tile_w = right - left
            weight_h = noise.new_ones(tile_h)
            weight_w = noise.new_ones(tile_w)

            if tile_overlap > 0:
                overlap_h = min(tile_overlap, tile_h - 1)
                overlap_w = min(tile_overlap, tile_w - 1)

                if overlap_h > 0:
                    ramp_h = torch.linspace(0, 1, overlap_h + 2, device=noise.device, dtype=noise.dtype)[1:-1]
                    if top > 0:
                        weight_h[:overlap_h] *= ramp_h
                    if bottom < height:
                        weight_h[-overlap_h:] *= ramp_h.flip(0)

                if overlap_w > 0:
                    ramp_w = torch.linspace(0, 1, overlap_w + 2, device=noise.device, dtype=noise.dtype)[1:-1]
                    if left > 0:
                        weight_w[:overlap_w] *= ramp_w
                    if right < width:
                        weight_w[-overlap_w:] *= ramp_w.flip(0)

            return weight_h.view(1, 1, tile_h, 1) * weight_w.view(1, 1, 1, tile_w)

        row_starts = get_tile_starts(height)
        col_starts = get_tile_starts(width)

        image_seq_len = height * width
        mu = self.compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        timesteps, num_inference_steps = self.retrieve_timesteps(
            scheduler,
            num_inference_steps=num_steps,
            device=noise.device,
            mu=mu,
        )

        t_init = self.noise_scheduler.safe_clamp(
            timesteps[0] / 1000.0, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t
        )
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        for timestep in timesteps:
            t = (timestep / 1000.0).expand(batch_size)
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            noise_pred_sum = torch.zeros_like(latents)
            weight_sum = torch.zeros(
                (batch_size, 1, height, width),
                device=latents.device,
                dtype=latents.dtype,
            )

            for top in row_starts:
                bottom = min(top + tile_size, height)
                for left in col_starts:
                    right = min(left + tile_size, width)
                    tile_latents = latents[:, :, top:bottom, left:right]
                    tile_condition = crop_condition(condition, top, left, bottom, right)

                    tile_noise_pred = self(
                        tile_latents,
                        t,
                        tile_condition,
                        fwd_pred_type="flow",
                        guidance=None,
                    )

                    if neg_condition is not None:
                        tile_neg_condition = crop_condition(neg_condition, top, left, bottom, right)
                        tile_noise_pred_uncond = self(
                            tile_latents,
                            t,
                            tile_neg_condition,
                            fwd_pred_type="flow",
                            guidance=None,
                        )
                        tile_noise_pred = tile_noise_pred_uncond + guidance_scale * (
                            tile_noise_pred - tile_noise_pred_uncond
                        )

                    tile_weight = get_tile_weight(top, left, bottom, right)
                    noise_pred_sum[:, :, top:bottom, left:right] += tile_noise_pred * tile_weight
                    weight_sum[:, :, top:bottom, left:right] += tile_weight

            noise_pred = noise_pred_sum / weight_sum.clamp_min(torch.finfo(latents.dtype).eps)
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents
