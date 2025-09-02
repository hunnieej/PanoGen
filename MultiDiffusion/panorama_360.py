from sched import scheduler
from transformers import (CLIPTextModel, 
                          CLIPTokenizer,
                          CLIPTextModelWithProjection, #text encoder_2 for SDXL
                          T5EncoderModel, #text encoder_3 for SD v3.5
                          T5Tokenizer, #tokenizer_3 for SD v3.5
                          logging,
                          Gemma2Model,
                          GemmaTokenizerFast)
from diffusers import (AutoencoderKL, 
                       UNet2DConditionModel, 
                       DDIMScheduler,
                       EulerDiscreteScheduler, #SDXL
                       DPMSolverMultistepScheduler, #SANA
                       FlowMatchEulerDiscreteScheduler, #SD v3.5
                       SanaTransformer2DModel,
                       SD3Transformer2DModel,
                       AutoencoderDC)
# TODO : 함수 정리
# functions : Sphere - S^2 - ERP 관련 함수 넣기
# utils : 그 외 다른 함수들 넣기
from projection import (generate_fibonacci_lattice, 
                   spherical_to_perspective_tiles, 
                   compute_patch_directions,
                   perspective_to_spherical_latent,
                   stitch_final_erp_from_tiles,
                   stitch_final_erp_from_tiles_xl
                   )

from utils import (build_2d_sincos_pe,
                   build_spherical_sincos_pe,
                    make_timestep_1d,
                    save_tiles_on_panorama
                    )

# suppress partial model loading warning
logging.set_verbosity_error()
import torch
import torch.nn as nn
import torchvision.transforms as T
import math, os, inspect, argparse, ast, copy
from tqdm import tqdm

########################################################################################################

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

########################################################################################################
# TODO : SANA, SDXL, SD v3.5 구현 
class SphereDiffusion(nn.Module):
    def __init__(self, device, rf_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.rf_version = rf_version

        print(f'[INFO] loading Reference diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.rf_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.rf_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.rf_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.rf_version =='xl':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        elif self.rf_version == '3.5':
            model_key = "stabilityai/stable-diffusion-3.5-medium"
        elif self.rf_version == 'SANA':
            # model_key= "Efficient-Large-Model/Sana_600M_512px_diffusers"
            model_key= "Efficient-Large-Model/Sana_600M_1024px_diffusers"

        else:
            raise ValueError(f'Reference diffusion version {self.rf_version} not supported.')

        # NOTE : version specific loading
        # DiT 기반
        if self.rf_version == 'SANA':
            self.vae = AutoencoderDC.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer = GemmaTokenizerFast.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = Gemma2Model.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.unet = SanaTransformer2DModel.from_pretrained(model_key, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler")
        elif self.rf_version == '3.5':
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2")
            self.tokenizer_3 = T5Tokenizer.from_pretrained(model_key, subfolder="tokenizer_3")
            self.text_encoder_1 = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_3 = T5EncoderModel.from_pretrained(model_key, subfolder="text_encoder_3", torch_dtype=torch.bfloat16).to(self.device)
            self.unet = SD3Transformer2DModel.from_pretrained(model_key, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler")
        # UNet 기반 
        elif self.rf_version == 'xl':
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(self.device)
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2")
            self.text_encoder_1 = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(self.device)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(self.device)
            self.scheduler = EulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler")
        else: #SD v1.5 / 2.0 / 2.1
            # dtype : float32
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self._supports_pos_embed = False
        print(f'[INFO] loaded reference diffusion!')
        
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]
        # Tokenize text and get embeddings
        # SD v1.5 / 2.0 / 2.1
        
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    @torch.no_grad()
    def get_text_embeds_xl(self, prompt, negative_prompt="", *,
                           prompt_2=None, negative_prompt_2=None,
                           max_length=None, height=1024, width=1024,
                           crop_coords=(0,0), target_size=None):
        """
        Returns:
        prompt_embeds           : [2B, L, C_total]  -> UNet(encoder_hidden_states=...)
        pooled_prompt_embeds    : [2B, D2]         -> UNet(added_cond_kwargs['text_embeds'])
        add_time_ids            : [2B, 6]          -> UNet(added_cond_kwargs['time_ids'])
        Notes:
        - Order is [uncond, cond] for CFG.
        - C_total = dim(text_encoder_2) + dim(text_encoder_1) (보통 1280+768)
        """
        # Normalize inputs to lists
        if isinstance(prompt, str): prompt = [prompt]
        B = len(prompt)

        if prompt_2 is None: prompt_2 = prompt
        if isinstance(prompt_2, str): prompt_2 = [prompt_2] * B

        if isinstance(negative_prompt, str): negative_prompt = [negative_prompt] * B
        if negative_prompt_2 is None: negative_prompt_2 = negative_prompt
        if isinstance(negative_prompt_2, str): negative_prompt_2 = [negative_prompt_2] * B

        tok1, tok2 = self.tokenizer_1, self.tokenizer_2
        te1,  te2  = self.text_encoder_1, self.text_encoder_2

        # pad token safety
        for tok in (tok1, tok2):
            if tok.pad_token is None:
                tok.pad_token = getattr(tok, "eos_token", tok.unk_token)
            tok.padding_side = "right"

        def _encode_pair(tokenizer, encoder, texts, maxlen):
            inputs = tokenizer(
                texts,
                padding="max_length",
                max_length=max_length or tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # For SDXL we want both sequence features and pooled features
            out = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                output_hidden_states=True,
            )
            seq = out.hidden_states[-2]             # [B, L, C*]
            # pooled: CLIPTextModel -> pooled_output, CLIPTextModelWithProjection -> text_embeds
            if hasattr(out, "pooled_output") and out.pooled_output is not None:
                pooled = out.pooled_output         # [B, D*]
            elif hasattr(out, "text_embeds") and out.text_embeds is not None:
                pooled = out.text_embeds           # [B, D*]
            else:
                pooled = seq[:, 0]                 # fallback: CLS
            return seq, pooled

        # cond
        seq1_c, pooled1_c = _encode_pair(tok1, te1, prompt,   max_length)
        seq2_c, pooled2_c = _encode_pair(tok2, te2, prompt_2, max_length)
        # uncond (negative)
        seq1_u, pooled1_u = _encode_pair(tok1, te1, negative_prompt,   max_length)
        seq2_u, pooled2_u = _encode_pair(tok2, te2, negative_prompt_2, max_length)

        # concat sequence features from both encoders (convention: encoder_2 || encoder_1)
        cond_seq = torch.cat([seq2_c, seq1_c], dim=-1)   # [B, L, C2+C1]
        uncond_seq = torch.cat([seq2_u, seq1_u], dim=-1) # [B, L, C2+C1]
        prompt_embeds = torch.cat([uncond_seq, cond_seq], dim=0).to(dtype=self.unet.dtype, device=self.device)  # [2B, L, Ctot]

        # pooled: use encoder_2 pooled (official pipeline behavior)
        cond_pooled = pooled2_c
        uncond_pooled = pooled2_u
        pooled_prompt_embeds = torch.cat([uncond_pooled, cond_pooled], dim=0).to(dtype=self.unet.dtype, device=self.device)  # [2B, D2]

        # micro-conditioning time ids
        if target_size is None:
            target_size = (height, width)
        add_time_ids = torch.tensor(
            [height, width, crop_coords[0], crop_coords[1], target_size[0], target_size[1]],
            device=self.device, dtype=self.unet.dtype
        ).repeat(prompt_embeds.shape[0], 1)  # [2B, 6]

        return prompt_embeds, pooled_prompt_embeds, add_time_ids

    @torch.no_grad()
    def get_text_embeds_sd35(self, prompt, negative_prompt="", max_sequence_length=77):
        # TODO : 구현해야함
        return 

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
        embeds = embeds.to(dtype=self.unet.dtype, device=self.device)
        attn_mask = torch.cat([uncond["attention_mask"], cond["attention_mask"]], dim=0) # [2B, L]
        return embeds, attn_mask

    ########################################################################################################

    @torch.no_grad()
    def text2panorama360(
        self, prompts, negative_prompts='',
        prompt_2=None, negative_prompt_2=None,
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
        dirs = generate_fibonacci_lattice(N_dirs).to(self.device, self.unet.dtype)  # [N,3]
        C = self.unet.config.in_channels if hasattr(self.unet, "config") else self.unet.in_channels
        feats = torch.randn(N_dirs, C, device=self.device, dtype=self.unet.dtype)   # global S_T

        # ----- tiles -----
        tiles = spherical_to_perspective_tiles(dirs=dirs, H=tile_h, W=tile_w, fov_deg=fov_deg, overlap=overlap)
        tau = 0.6 * math.tan(math.radians(fov_deg) * 0.5)  # distortion-aware weight
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
                    embeds=embeds.to(self.unet.dtype).to(self.device),
                    attn=attn
                )
            elif self.rf_version == 'xl':
                pe, pooled, time_ids = self.get_text_embeds_xl(prompts, negative_prompts)
                return dict(
                    kind='xl',
                    prompt_embeds=pe.to(self.unet.dtype).to(self.device),
                    pooled=pooled.to(self.unet.dtype).to(self.device),
                    time_ids=time_ids.to(self.unet.dtype).to(self.device)
                )
            elif self.rf_version == '3.5':
                pe, pooled = self.get_text_embeds_sd35(prompts, negative_prompts)
                return dict(
                    kind='sd35',
                    prompt_embeds=pe.to(self.unet.dtype).to(self.device),
                    pooled=pooled.to(self.unet.dtype).to(self.device)
                )
            else:  # 1.5 / 2.0 / 2.1
                embeds = self.get_text_embeds(prompts, negative_prompts)
                return dict(
                    kind='sd',
                    embeds=embeds.to(self.unet.dtype).to(self.device)
                )

        text_cond = _prepare_text_cond()

        # ----- pos-embed meta (DiT-like) -----
        if self.rf_version in ['SANA','3.5']:
            patch = getattr(self.unet, "patch_size", 2)
            Dm = getattr(self.unet, "inner_dim",
                getattr(self.unet, "hidden_size",
                getattr(getattr(self.unet, "config", object()), "inner_dim", 1024)))
        else:
            patch, Dm = None, None

        # ===== helpers =====
        def _make_cond_kwargs(batch_size):
            if text_cond['kind'] == 'sana':
                return dict(encoder_hidden_states=text_cond['embeds'],
                            encoder_attention_mask=text_cond['attn'])
            elif text_cond['kind'] == 'sd35':
                return dict(
                    encoder_hidden_states=text_cond['prompt_embeds'],
                    added_cond_kwargs=dict(text_embeds=text_cond['pooled'])
                )
            elif text_cond['kind'] == 'xl':
                return dict(encoder_hidden_states=text_cond['prompt_embeds'],
                            added_cond_kwargs=dict(text_embeds=text_cond['pooled'],
                                                time_ids=text_cond['time_ids']))
            else:
                return dict(encoder_hidden_states=text_cond['embeds'])

        def _unet_forward(latent_in, step_idx, t_scalar, cond_kwargs):
            if self.rf_version =='SANA':
                lin = self.scheduler.scale_model_input(latent_in, t_scalar)
                t_in = make_timestep_1d(t_scalar, latent_in.shape[0], self.device, float_for_dit=True)
                out = self.unet(hidden_states=lin, timestep=t_in, **cond_kwargs, return_dict=True).sample
            elif self.rf_version == '3.5':
                lin = self.scheduler.scale_model_input(latent_in, t_scalar)
                t_in = make_timestep_1d(t_scalar, latent_in.shape[0], self.device, float_for_dit=True)
                out = self.unet(hidden_states=lin, timestep=t_in, **cond_kwargs, return_dict=True).sample
            elif self.rf_version == 'xl':
                t_model = self.scheduler.timesteps[step_idx]
                lin = self.scheduler.scale_model_input(latent_in, t_model)
                out = self.unet(sample=lin, timestep=t_model, **cond_kwargs, return_dict=True).sample
            else:
                out = self.unet(sample=latent_in, timestep=t_scalar, **cond_kwargs, return_dict=True).sample
            return out

        def _apply_cfg(noise_2B):
            if not use_cfg: return noise_2B
            n_u, n_c = noise_2B.chunk(2, dim=0)
            return n_u + guidance_scale * (n_c - n_u)

        def _gather_tile(feats_global, tile, C):
            used_idx = tile["used_idx"].long()
            lin = tile["lin_coords"].long()
            Ht, Wt = tile["H"], tile["W"]
            flat = torch.zeros(Ht * Wt, C, device=self.device, dtype=self.unet.dtype)
            flat.index_copy_(0, lin, feats_global[used_idx])
            return flat.view(Ht, Wt, C).permute(2,0,1).unsqueeze(0)  # [1,C,Ht,Wt]

        def _save_tile_preview_if_needed(tile_eps, I_bchw, step_idx, tile_idx):
            if not save_tile_intermediate or (step_idx % 49 != 0): return
            tmp_sched = copy.deepcopy(self.scheduler)
            t_prev = tmp_sched.timesteps[step_idx] if self.rf_version == 'xl' else self.scheduler.timesteps[step_idx]
            step_res = tmp_sched.step(model_output=tile_eps.unsqueeze(0), timestep=t_prev, sample=I_bchw)
            xprev = step_res.prev_sample
            if xprev.dim() == 5 and xprev.shape[1] == 1: xprev = xprev.squeeze(1)
            if xprev.dim() == 3: xprev = xprev.unsqueeze(0)

            if self.rf_version == 'xl':
                scale, autocast_enabled, vae_dtype_ctx = 1/0.13025, False, torch.float32
            elif self.rf_version =='SANA':
                scale, autocast_enabled, vae_dtype_ctx = 1/0.41407, True, torch.bfloat16
            elif self.rf_version == '3.5':
                scale, autocast_enabled, vae_dtype_ctx = 1/1.5305, True, torch.bfloat16
            else:
                scale, autocast_enabled, vae_dtype_ctx = 1/0.18215, True, torch.bfloat16

            lat = (xprev * scale).to(torch.float32)
            if self.rf_version in ['1.5','2.0','2.1','xl'] :
                if getattr(self.vae.config, "force_upcast", False):
                    self.vae.to(dtype=torch.float32)
                vae_in_dtype = next(self.vae.post_quant_conv.parameters()).dtype
            else:
                vae_in_dtype = torch.bfloat16
            lat = lat.to(device=self.vae.device, dtype=vae_in_dtype)
            with torch.autocast(device_type='cuda', enabled=autocast_enabled, dtype=vae_dtype_ctx):
                dec = self.vae.decode(lat).sample
            tile_img = (dec[0].float()/2 + 0.5).clamp(0,1)
            T.ToPILImage()(tile_img.cpu()).save(os.path.join(outfolder, f"tile_{tile_idx:03d}_{step_idx:03d}.png"))

        def _decode_erp_from_feats(feats_spherical):
            if self.rf_version == 'xl':
                erp = stitch_final_erp_from_tiles_xl(
                    feats_spherical=feats_spherical, dirs=dirs, tiles=tiles, vae=self.vae,
                    scale=1/0.13025, pano_H=H, pano_W=W, fov_deg=fov_deg
                )
            else:
                erp = stitch_final_erp_from_tiles(
                    feats_spherical=feats_spherical, dirs=dirs, tiles=tiles, vae=self.vae,
                    scale=(1/0.41407 if self.rf_version=='SANA' else 1/1.5305 if self.rf_version=='3.5' else 1/0.18215),
                    pano_H=H, pano_W=W, fov_deg=fov_deg, rf_version=self.rf_version
                )
            return erp  # [3,H,W] in [0,1]

        def _save_erp_if_needed(feats_spherical, step_idx):
            if not save_intermediate: return
            if step_idx % 10 != 0 and step_idx != (len(self.scheduler.timesteps)-1): return
            erp_mid = _decode_erp_from_feats(feats_spherical)
            T.ToPILImage()(erp_mid.cpu()).save(os.path.join(outfolder, f"step_{step_idx:03d}.png"))

        # ===== diffusion loop =====
        self.scheduler.set_timesteps(num_inference_steps)
        total_steps = len(self.scheduler.timesteps)
        print(f"[INFO] Output folder ready: {outfolder}")
        if save_intermediate:
            print(f"[INFO] Will save {len(tiles)} tiles × {num_inference_steps} steps = {len(tiles)*num_inference_steps} tile images")

        with tqdm(total=total_steps, desc='Generating panorama') as pbar:
            for step_idx, t_scalar in enumerate(self.scheduler.timesteps):
                tile_eps_list = []
                for tile_idx, tile in enumerate(tiles):
                    I = _gather_tile(feats, tile, C)                      # [1,C,Ht,Wt]
                    latent_in = torch.cat([I, I], dim=0) if use_cfg else I
                    cond_kwargs = _make_cond_kwargs(latent_in.shape[0])
                    noise = _unet_forward(latent_in, step_idx, t_scalar, cond_kwargs)  # [B,C,Ht,Wt]

                    eps_tile = _apply_cfg(noise) if use_cfg else noise                 # [1,C,Ht,Wt]
                    tile_eps_list.append(eps_tile.squeeze(0))                          # [C,Ht,Wt]
                    _save_tile_preview_if_needed(eps_tile.squeeze(0), I, step_idx, tile_idx)

                eps_tiles = torch.stack(tile_eps_list, dim=0)  # [T,C,Ht,Wt]
                fused_eps = perspective_to_spherical_latent(
                    perspective_feats=eps_tiles, tiles_info=tiles, sphere_dirs=dirs, tau=tau
                )  # [N_dirs, C]
                feats = self.scheduler.step(model_output=fused_eps, timestep=t_scalar, sample=feats).prev_sample

                # ---- save mid ERP (명시적 호출) ----
                _save_erp_if_needed(feats, step_idx)

                pbar.update(1)

        # ===== final decode =====
        erp_img = _decode_erp_from_feats(feats)
        return T.ToPILImage()(erp_img.cpu())
    
########################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='a photo of the dolomites')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--rf_version', type=str, default='2.0', choices=['1.5', '2.0', '2.1','3.5', 'xl', 'SANA'],
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

    seed_everything(opt.seed)

    device = torch.device(f'cuda:{opt.gpu}')

    sd = SphereDiffusion(device, opt.rf_version)

    img = sd.text2panorama360(opt.prompt, opt.negative,
                              opt.prompt, opt.negative, # SDXL
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
