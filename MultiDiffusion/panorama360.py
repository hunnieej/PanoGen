from sched import scheduler
from transformers import (CLIPTokenizer,
                          CLIPTextModelWithProjection, #text encoder_2 for SDXL
                          T5EncoderModel, #text encoder_3 for SD v3.5
                          T5Tokenizer, #tokenizer_3 for SD v3.5
                          logging,
                          Gemma2Model,
                          GemmaTokenizerFast)
from diffusers import (AutoencoderKL, 
                       DPMSolverMultistepScheduler, #SANA
                       FlowMatchEulerDiscreteScheduler, #SD v3.5
                       SanaTransformer2DModel,
                       SD3Transformer2DModel,
                       AutoencoderDC)

# functions : Sphere - S^2 - ERP 관련 함수 넣기
# utils : 그 외 다른 함수들 넣기
from projection360 import (generate_fibonacci_lattice, 
                   spherical_to_perspective_tiles,
                   log_overlap_between_views,
                   compute_and_save_overlap_matrix,
                   fuse_perspective_to_spherical
                   )
from projection_erp import (stitch_final_erp_from_tiles)

from utils import (make_timestep_1d,
                   save_tiles_on_panorama
                   )

# suppress partial model loading warning
logging.set_verbosity_error()
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import math, os, argparse, ast, copy, yaml, csv, json
from tqdm import tqdm
import numpy as np

########################################################################################################

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

########################################################################################################
class SphereDiffusion(nn.Module):
    def __init__(self, device, rf_version='3.5', hf_key=None):
        super().__init__()

        self.device = device
        self.rf_version = rf_version

        print(f'[INFO] loading Reference diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.rf_version == '3.5':
            model_key = "stabilityai/stable-diffusion-3.5-medium"
        elif self.rf_version == 'SANA':
            model_key= "Efficient-Large-Model/Sana_600M_1024px_diffusers"
        else:
            raise ValueError(f'Reference diffusion version {self.rf_version} not supported.')

        # NOTE : version specific loading
        # DiT 기반
        if self.rf_version == 'SANA':
            self.vae = AutoencoderDC.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer = GemmaTokenizerFast.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = Gemma2Model.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.transformer = SanaTransformer2DModel.from_pretrained(model_key, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler")
        else: # self.rf_version == '3.5'
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2")
            self.tokenizer_3 = T5Tokenizer.from_pretrained(model_key, subfolder="tokenizer_3")
            self.text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_3 = T5EncoderModel.from_pretrained(model_key, subfolder="text_encoder_3", torch_dtype=torch.bfloat16).to(self.device)
            self.transformer = SD3Transformer2DModel.from_pretrained(model_key, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded reference diffusion!')

    @torch.no_grad()
    def get_text_embeds_sd35(
        self,
        prompt,
        negative_prompt="",
        max_sequence_length=512,
        clip_skip=None,
        device=None,
        dtype=None
    ):

        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * B
        elif len(negative_prompt) != B:
            raise ValueError(f"negative_prompt length ({len(negative_prompt)}) must match prompt length ({B})")

        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.transformer.dtype

        tok1, tok2, tok3 = self.tokenizer_1, self.tokenizer_2, self.tokenizer_3
        te1, te2, te3 = self.text_encoder_1, self.text_encoder_2, self.text_encoder_3

        for tok in (tok1, tok2):
            if tok.pad_token is None:
                tok.pad_token = getattr(tok, "eos_token", tok.unk_token)
            tok.padding_side = "right"

        d_clip1 = te1.config.hidden_size #768
        d_clip2 = te2.config.hidden_size #1280
        d_t5 = te3.config.d_model #4096
        joint_dim = getattr(self.transformer.config, "joint_attention_dim", d_t5) #4096

        print(f"Text Encoder dim: CLIP1={d_clip1}, CLIP2={d_clip2}, T5={d_t5}, Joint={joint_dim}")

        def encode_clip(tokenizer, encoder, texts, skip_layers=None):
            try:
                inputs = tokenizer(
                    texts, 
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, 
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = encoder(
                        input_ids=inputs["input_ids"].to(device), 
                        output_hidden_states=True
                    )

                if skip_layers is None:
                    hidden_states = outputs.hidden_states[-2]
                else:
                    layer_idx = -(skip_layers + 2)
                    hidden_states = outputs.hidden_states[layer_idx]

                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    pooled = outputs.pooler_output
                elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                    pooled = outputs.text_embeds
                else:
                    pooled = hidden_states[:, 0]
                    
                return hidden_states.to(dtype), pooled.to(dtype)
                
            except Exception as e:
                raise RuntimeError(f"CLIP Encoding Failed: {e}")

        def encode_t5(tokenizer, encoder, texts):
            try:
                inputs = tokenizer(
                    texts, 
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True, 
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = encoder(input_ids=inputs["input_ids"].to(device))
                    
                return outputs.last_hidden_state.to(dtype)
                
            except Exception as e:
                raise RuntimeError(f"T5 Encoding Failed : {e}")

        def project_to_joint_dim(embeddings, source_dim, target_dim, name):
            if source_dim == target_dim:
                return embeddings
            else:
                pad_size = target_dim - source_dim
                return torch.nn.functional.pad(embeddings, (0, pad_size))

        clip1_cond, pooled1_cond = encode_clip(tok1, te1, prompt, clip_skip)
        clip2_cond, pooled2_cond = encode_clip(tok2, te2, prompt, clip_skip)
        t5_cond = encode_t5(tok3, te3, prompt)
        
        clip1_uncond, pooled1_uncond = encode_clip(tok1, te1, negative_prompt, clip_skip)
        clip2_uncond, pooled2_uncond = encode_clip(tok2, te2, negative_prompt, clip_skip)
        t5_uncond = encode_t5(tok3, te3, negative_prompt)

        clip1_cond = project_to_joint_dim(clip1_cond, d_clip1, joint_dim, "clip1")
        clip2_cond = project_to_joint_dim(clip2_cond, d_clip2, joint_dim, "clip2")
        t5_cond = project_to_joint_dim(t5_cond, d_t5, joint_dim, "t5")
        
        clip1_uncond = project_to_joint_dim(clip1_uncond, d_clip1, joint_dim, "clip1")
        clip2_uncond = project_to_joint_dim(clip2_uncond, d_clip2, joint_dim, "clip2")
        t5_uncond = project_to_joint_dim(t5_uncond, d_t5, joint_dim, "t5")

        cond_embeds = torch.cat([clip1_cond, clip2_cond, t5_cond], dim=1)
        uncond_embeds = torch.cat([clip1_uncond, clip2_uncond, t5_uncond], dim=1)

        prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)

        pooled_cond = torch.cat([pooled1_cond, pooled2_cond], dim=-1)
        pooled_uncond = torch.cat([pooled1_uncond, pooled2_uncond], dim=-1)
        pooled_proj = torch.cat([pooled_uncond, pooled_cond], dim=0)

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_proj = pooled_proj.to(device=device, dtype=dtype)
        
        print(f"Done - prompt_embeds: {prompt_embeds.shape}, pooled_proj: {pooled_proj.shape}")
        
        return prompt_embeds, pooled_proj

    @torch.no_grad()
    def get_text_embeds_sana(self, prompt, negative_prompt="", max_sequence_length=300):
        """
        Args:
        - prompt, negative_prompt: str or List[str]
        Returns:
        - embeds: [2B, L, C], attention_mask: [2B, L]
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", self.tokenizer.unk_token)
        self.tokenizer.padding_side = "right"

        common = dict(
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        cond = self.tokenizer(prompt, **common)
        uncond = self.tokenizer(negative_prompt, **common)
        dev = self.device
        cond = {k: v.to(dev) for k, v in cond.items()}
        uncond = {k: v.to(dev) for k, v in uncond.items()}

        cond_out = self.text_encoder(
            input_ids=cond["input_ids"],
            attention_mask=cond["attention_mask"],
        )
        uncond_out = self.text_encoder(
            input_ids=uncond["input_ids"],
            attention_mask=uncond["attention_mask"],
        )

        cond_emb = cond_out.last_hidden_state
        uncond_emb = uncond_out.last_hidden_state

        embeds = torch.cat([uncond_emb, cond_emb], dim=0)         # [2B, L, C]
        embeds = embeds.to(dtype=self.transformer.dtype, device=self.device)
        attn_mask = torch.cat([uncond["attention_mask"], cond["attention_mask"]], dim=0) # [2B, L]
        return embeds, attn_mask

    ########################################################################################################

    @torch.no_grad()
    def text2panorama360(
        self, prompts, negative_prompts='',
        H=512, W=4096, num_inference_steps=50, guidance_scale=7.5,
        fov_deg=80.0, overlap=0.6, N_dirs=2600, tile_h=64, tile_w=64,
        use_cfg=True, outfolder='out',
        save_intermediate=False, save_tile_intermediate=False, save_tile_panorama=False
    ):
        # ===== Common setup =====
        self.fov_deg, self.overlap, self.N_dirs = fov_deg, overlap, N_dirs
        self.num_inference_steps, self.use_cfg = num_inference_steps, use_cfg
        self.guidance_scale, self.H_lat, self.W_lat = guidance_scale, tile_h, tile_w
        self.H, self.W = H, W
        os.makedirs(outfolder, exist_ok=True)
        print(f'[INFO] Generating panorama rf={self.rf_version} | N_dirs={N_dirs}, FOV={fov_deg}°, overlap={overlap}, tile={tile_h}x{tile_w}')

        # ----- text embeds (once) -----
        if isinstance(prompts, str): prompts = [prompts]
        if isinstance(negative_prompts, str): negative_prompts = [negative_prompts]

        # ----- spherical directions & global latent -----
        dirs = generate_fibonacci_lattice(N_dirs).to(self.device, self.transformer.dtype)  # [N,3]
        C = self.transformer.config.in_channels if hasattr(self.transformer, "config") else self.transformer.in_channels
        feats = torch.randn(N_dirs, C, device=self.device, dtype=self.transformer.dtype)   # [N, C]

        # ----- tiles -----
        tiles = spherical_to_perspective_tiles(dirs=dirs, 
                                               Hp=tile_h, Wp=tile_w, 
                                               fov_deg=fov_deg)
        log_path = os.path.join(outfolder, "tile_info")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        tile_info_path = os.path.join(log_path, "tile_info.txt")
        tile_csv_path = os.path.join(log_path, "tile_info.csv")
        os.makedirs(os.path.dirname(tile_info_path), exist_ok=True)
        os.makedirs(os.path.dirname(tile_csv_path), exist_ok=True)
        log_overlap_between_views(tiles, log_path=tile_info_path)
        compute_and_save_overlap_matrix(tiles, log_path=tile_csv_path, mode="count")
        
        tau = 0.6 * math.tan(math.radians(fov_deg) * 0.5)  # FoV 80 일때 0.5 정도 나옴
        if save_tile_panorama:
            save_tiles_on_panorama(tiles, pano_H=H, pano_W=W,
                                    mode="outline", step=6,
                                    outpath=os.path.join(outfolder, f"tile_on_pano_outline_{N_dirs}.png"),
                                    annotate=True, save_groups=True, group_size=10)

        def _prepare_text_cond():
            if self.rf_version == 'SANA':
                embeds, attn = self.get_text_embeds_sana(prompts, negative_prompts)
                return dict(
                    kind='sana',
                    embeds=embeds.to(self.transformer.dtype).to(self.device),
                    attn=attn
                )
            else:  # self.rf_version == '3.5':
                pe, pooled = self.get_text_embeds_sd35(prompts, negative_prompts)
                return dict(
                    kind='sd35',
                    prompt_embeds=pe.to(self.transformer.dtype).to(self.device),
                    pooled=pooled.to(self.transformer.dtype).to(self.device)
                )

        text_cond = _prepare_text_cond()

        # ===== helpers =====
        def _make_cond_kwargs(batch_size):
            if text_cond['kind'] == 'sana':
                return dict(encoder_hidden_states=text_cond['embeds'],
                            encoder_attention_mask=text_cond['attn'])
            elif text_cond['kind'] == 'sd35':
                return dict(
                    encoder_hidden_states=text_cond['prompt_embeds'],
                    # pooled_projections=dict(text_embeds=text_cond['pooled'])
                    pooled_projections=text_cond['pooled']
                )
            else:
                return dict(encoder_hidden_states=text_cond['embeds'])

        def _transformer_forward(latent_in, t_scalar, cond_kwargs):
            if self.rf_version =='SANA':
                lin = self.scheduler.scale_model_input(latent_in, t_scalar)
                t_in = make_timestep_1d(t_scalar, latent_in.shape[0], self.device, float_for_dit=True)
                out = self.transformer(hidden_states=lin, timestep=t_in, **cond_kwargs, return_dict=True).sample
            else:  # self.rf_version == '3.5':
                lin = self.scheduler.scale_model_input(latent_in, t_scalar) \
                    if hasattr(self.scheduler, "scale_model_input") else latent_in
                t_in = make_timestep_1d(t_scalar, latent_in.shape[0], self.device, float_for_dit=True)
                out = self.transformer(
                    hidden_states=lin,
                    timestep=t_in,
                    **cond_kwargs,
                    return_dict=True
                ).sample
            return out

        def _apply_cfg(noise_2B):
            if not use_cfg: return noise_2B
            n_u, n_c = noise_2B.chunk(2, dim=0)
            return n_u + guidance_scale * (n_c - n_u)

        def _gather_tile(feats_global, tile, C):
            used_idx = tile["used_idx"].long()
            lin      = tile["lin_coords"].long()
            perm = torch.argsort(lin)
            lin = lin[perm]; used_idx = used_idx[perm]
            Hp, Wp   = int(tile["H"]), int(tile["W"])
            flat = torch.zeros(Hp * Wp, C, device=self.device, dtype=self.transformer.dtype)
            flat.index_copy_(0, lin, feats_global[used_idx])
            return flat.view(Hp, Wp, C).permute(2, 0, 1).unsqueeze(0)

        # def _save_tile_preview_if_needed(tile_eps, I_bchw, step_idx, tile_idx):
        #     if not save_tile_intermediate or not(step_idx % 10 == 0 or step_idx % 49 == 0):
        #         return

        #     tmp_sched = copy.deepcopy(self.scheduler)
        #     t_prev = tmp_sched.timesteps[step_idx]
        #     step_res = tmp_sched.step(model_output=tile_eps.unsqueeze(0), timestep=t_prev, sample=I_bchw)
        #     xprev = step_res.prev_sample
        #     if xprev.dim() == 5 and xprev.shape[1] == 1:
        #         xprev = xprev.squeeze(1)
        #     if xprev.dim() == 3:
        #         xprev = xprev.unsqueeze(0)

        #     sf = float(getattr(self.vae.config, "scaling_factor", 1.0))
        #     sh = float(getattr(self.vae.config, "shift_factor", 0.0))

        #     if self.rf_version == 'SANA':
        #         autocast_enabled = True
        #         vae_in_dtype = torch.bfloat16
        #         lat = (xprev / 0.41407).to(torch.float32) + 0.0

        #     else:  # self.rf_version == '3.5'
        #         force_upcast = bool(getattr(self.vae.config, "force_upcast", False))
        #         if force_upcast:
        #             self.vae.to(dtype=torch.float32)
        #             vae_in_dtype = torch.float32
        #             autocast_enabled = False
        #         else:
        #             vae_in_dtype = getattr(self.vae, "dtype", torch.bfloat16)
        #             autocast_enabled = (vae_in_dtype in (torch.bfloat16, torch.float16))

        #         lat = (xprev.to(torch.float32) / sf) + sh

        #     lat = lat.to(device=self.vae.device, dtype=vae_in_dtype)

        #     if autocast_enabled and vae_in_dtype != torch.float32:
        #         with torch.autocast(device_type='cuda', enabled=True, dtype=vae_in_dtype):
        #             dec = self.vae.decode(lat).sample  # [-1,1]
        #     else:
        #         dec = self.vae.decode(lat).sample
 
        #     tile_img = (dec[0].float() / 2 + 0.5).clamp(0, 1)
        #     tile_path = os.path.join(outfolder, "tile_intermediate")
        #     os.makedirs(tile_path, exist_ok=True)
        #     T.ToPILImage()(tile_img.cpu()).save(os.path.join(tile_path, f"tile_{tile_idx:03d}_{step_idx:03d}.png"))

        def _save_tile_preview_if_needed(tile_eps, I_bchw, step_idx, tile_idx, after_fusion: bool = False):
            if not save_tile_intermediate or not (step_idx % 10 == 0 or step_idx % 49 == 0):
                return

            if after_fusion:
                # 합성(전역 갱신) 이후의 타일 latent(I_bchw)를 그대로 디코드
                xprev = I_bchw
                if xprev.dim() == 3:
                    xprev = xprev.unsqueeze(0)          # [1,C,H,W]
            else:
                # 기존 동작: 미리보기용으로 타일만 임시 한 스텝 진행
                tmp_sched = copy.deepcopy(self.scheduler)
                t_prev = tmp_sched.timesteps[step_idx]
                step_res = tmp_sched.step(model_output=tile_eps.unsqueeze(0), timestep=t_prev, sample=I_bchw)
                xprev = step_res.prev_sample
                if xprev.dim() == 5 and xprev.shape[1] == 1:
                    xprev = xprev.squeeze(1)
                if xprev.dim() == 3:
                    xprev = xprev.unsqueeze(0)

            sf = float(getattr(self.vae.config, "scaling_factor", 1.0))
            sh = float(getattr(self.vae.config, "shift_factor", 0.0))

            if self.rf_version == 'SANA':
                autocast_enabled = True
                vae_in_dtype = torch.bfloat16
                lat = (xprev / 0.41407).to(torch.float32) + 0.0
            else:  # self.rf_version == '3.5'
                force_upcast = bool(getattr(self.vae.config, "force_upcast", False))
                if force_upcast:
                    self.vae.to(dtype=torch.float32)
                    vae_in_dtype = torch.float32
                    autocast_enabled = False
                else:
                    vae_in_dtype = getattr(self.vae, "dtype", torch.bfloat16)
                    autocast_enabled = (vae_in_dtype in (torch.bfloat16, torch.float16))
                lat = (xprev.to(torch.float32) / sf) + sh

            lat = lat.to(device=self.vae.device, dtype=vae_in_dtype)

            if autocast_enabled and vae_in_dtype != torch.float32:
                with torch.autocast(device_type='cuda', enabled=True, dtype=vae_in_dtype):
                    dec = self.vae.decode(lat).sample  # [-1,1]
            else:
                dec = self.vae.decode(lat).sample

            tile_img = (dec[0].float() / 2 + 0.5).clamp(0, 1)
            tile_path = os.path.join(outfolder, "tile_intermediate")
            os.makedirs(tile_path, exist_ok=True)
            suffix = "post" if after_fusion else "pre"
            T.ToPILImage()(tile_img.cpu()).save(os.path.join(tile_path, f"tile_{tile_idx:03d}_{step_idx:03d}_{suffix}.png"))

        def _decode_erp_from_feats(feats_spherical):
            erp = stitch_final_erp_from_tiles(
                feats_spherical=feats_spherical, tiles=tiles, vae=self.vae,
                pano_H=H, pano_W=W, fov_deg=fov_deg, rf_version=self.rf_version)
            return erp  # [3,H,W] in [0,1]

        def _save_erp_if_needed(feats_spherical, step_idx):
            if not save_intermediate: return
            if step_idx % 10 != 0 and step_idx != (len(self.scheduler.timesteps)-1): return
            erp_mid = _decode_erp_from_feats(feats_spherical)
            T.ToPILImage()(erp_mid.cpu()).save(os.path.join(outfolder, f"step_{step_idx:03d}.png"))

        def _one_step_with_tmp_scheduler(model_output, sample, step_idx):
            ts_full = self.scheduler.timesteps
            k0 = int(step_idx)
            if k0 >= len(ts_full) - 1:
                return sample
            k1 = k0 + 1

            tmp = type(self.scheduler).from_config(self.scheduler.config)
            tmp.set_timesteps(self.num_inference_steps)

            try:
                tmp.timesteps = ts_full.new_tensor([ts_full[k0], ts_full[k1]])
            except Exception:
                pass

            try:
                if hasattr(self.scheduler, "sigmas") and hasattr(tmp, "sigmas"):
                    tmp.sigmas = self.scheduler.sigmas[k0:k1+1]
            except Exception:
                pass

            step_res = tmp.step(
                model_output=model_output,
                timestep=tmp.timesteps[0],
                sample=sample,
            )
            return step_res.prev_sample

        self.scheduler.set_timesteps(num_inference_steps)
        total_steps = len(self.scheduler.timesteps)
        print(f"[INFO] Output folder ready: {outfolder}")
        if save_intermediate:
            print(f"[INFO] Will save {len(tiles)} tiles × {num_inference_steps} steps = {len(tiles)*num_inference_steps} tile images")
        with tqdm(total=total_steps, desc='Generating panorama') as pbar:
            for step_idx, t_scalar in enumerate(self.scheduler.timesteps):
                accum = torch.zeros_like(feats, dtype=torch.float32)            # [N_dirs,C]
                wsum  = torch.zeros(feats.shape[0], 1, device=self.device, dtype=torch.float32)

                for view_idx, tile in enumerate(tiles):
                    I = _gather_tile(feats, tile, C)
                    latent_in  = torch.cat([I, I], dim=0) if use_cfg else I
                    cond_kwargs = _make_cond_kwargs(latent_in.shape[0])
                    eps_bchw = _transformer_forward(latent_in, t_scalar, cond_kwargs)
                    eps_tile = _apply_cfg(eps_bchw) if use_cfg else eps_bchw
                    eps_tile = eps_tile.squeeze(0)  # [C,Ht,Wt]

                    tile_prev = _one_step_with_tmp_scheduler(
                        model_output=eps_tile.unsqueeze(0),
                        sample=I,
                        step_idx=step_idx
                    ).squeeze(0)  # [C,H,W]

                    acc_w, w_i = fuse_perspective_to_spherical(
                        tile_prev.unsqueeze(0), [tile], sphere_dirs=dirs,tau=tau
                    )
                    accum += acc_w
                    wsum  += w_i
                    _save_tile_preview_if_needed(eps_tile, I, step_idx, view_idx)
                feats = (accum / torch.clamp_min(wsum, 1e-6)).to(self.transformer.dtype)  # [N_dirs,C]

                _save_erp_if_needed(feats, step_idx)
                pbar.update(1)

        erp_img = _decode_erp_from_feats(feats)
        return T.ToPILImage()(erp_img.cpu())
    
########################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='a photo of the dolomites')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--rf_version', type=str, default='3.5', choices=['3.5', 'SANA'],
                        help="Reference diffusion version")
    parser.add_argument('--H', type=int, default=512, help="Panorama height")
    parser.add_argument('--W', type=int, default=4096, help="Panorama width")
    parser.add_argument('--Hlat', type=int, default=64, help="latent height")
    parser.add_argument('--Wlat', type=int, default=64, help="latent width")
    parser.add_argument('--fov_deg', type=float, default=80.0, help="Field of view in degrees")
    parser.add_argument('--overlap', type=float, default=0.6, help="Overlap between tiles as a fraction of FOV")
    parser.add_argument('--N_dirs', type=int, default=2600, help="Number of directions to sample")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--outfolder', type=str, default='out', help='Output folder for saving all images')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for classifier-free guidance')
    parser.add_argument('--save_intermediate', type=ast.literal_eval, choices = [True, False], default=False, help='Save intermediate results for each timestep')
    parser.add_argument('--save_tile_intermediate', type=ast.literal_eval, choices = [True, False], default=False, help='Save intermediate tile images for each step')
    parser.add_argument('--save_tile_panorama', type=ast.literal_eval, choices = [True, False], default=False, help='Save intermediate tile images in panorama layout')
    opt = parser.parse_args()

    # Save argparse arguments to yaml file in output folder
    args_dict = vars(opt)
    os.makedirs(opt.outfolder, exist_ok=True)
    yaml_filename = os.path.join(opt.outfolder, f"{os.path.basename(opt.outfolder)}.yaml")
    with open(yaml_filename, 'w') as f:
        yaml.dump(args_dict, f)
    print(f"[INFO] Saved experiment config to: {yaml_filename}")

    seed_everything(opt.seed)

    device = torch.device(f'cuda:{opt.gpu}')

    sd = SphereDiffusion(device, opt.rf_version)

    img = sd.text2panorama360(opt.prompt, opt.negative,
                              opt.H, opt.W, 
                              opt.steps, opt.guidance_scale, 
                              opt.fov_deg, opt.overlap, opt.N_dirs, opt.Hlat, opt.Wlat, 
                              outfolder=opt.outfolder, 
                              save_intermediate=opt.save_intermediate,
                              save_tile_intermediate=opt.save_tile_intermediate,
                              save_tile_panorama=opt.save_tile_panorama
                              )

    final_filename = os.path.join(opt.outfolder, "final.png")
    img.save(final_filename)
    print(f"[INFO] Saved final result: {final_filename}")

######################################################################################################## 