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

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    print(f'[INFO] Getting views for panorama')
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

def generate_fibonacci_lattice(N):
    # https://observablehq.com/@meetamit/fibonacci-lattices
    # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    print(f'[INFO] Generating fibonacci lattice with {N} points')
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    indices = torch.arange(N, dtype=torch.float32)
    z = 1 - 2 * indices / (N - 1)  # Uniform distribution in z: [1, -1]
    theta = 2 * np.pi * indices / phi
    rho = torch.sqrt(1 - z**2)  # Radius at height z
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    coords = torch.stack([x, y, z], dim=1)  # Shape: (N, 3)
    coords = coords / torch.norm(coords, dim=1, keepdim=True)
    
    return coords



class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading Reference diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version =='xl':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        elif self.sd_version == '3.5':
            model_key = "stabilityai/stable-diffusion-3.5-medium"
        elif self.sd_version == 'SANA':
            model_key= "Efficient-Large-Model/Sana_600M_512px_diffusers"
            # model_key = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
        else:
            raise ValueError(f'Reference diffusion version {self.sd_version} not supported.')

        # NOTE : version specific loading
        if self.sd_version == 'SANA':
            self.vae = AutoencoderDC.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer = GemmaTokenizerFast.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = Gemma2Model.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.unet = SanaTransformer2DModel.from_pretrained(model_key, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler")
        elif self.sd_version == 'xl':
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2")
            self.text_encoder_1 = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = EulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler")
        elif self.sd_version == '3.5':
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=torch.bfloat16).to(self.device)
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer_2")
            self.tokenizer_3 = T5Tokenizer.from_pretrained(model_key, subfolder="tokenizer_3")
            self.text_encoder_1 = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(self.device)
            self.text_encoder_3 = T5EncoderModel.from_pretrained(model_key, subfolder="text_encoder_3", torch_dtype=torch.bfloat16).to(self.device)
            self.unet = SD3Transformer2DModel.from_pretrained(model_key, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler")
        else: #SD v1.5 / 2.0 / 2.1
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

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
    def get_sana_text_embeds(self, prompt, negative_prompt="", max_sequence_length=192):
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

    @torch.no_grad()
    # NOTE : SD v3.5 check 필요
    def decode_latents(self, latents):
        if self.sd_version == 'SANA':
            latents = latents
        elif self.sd_version == 'xl':
            latents = 1 / 0.13025 * latents
        else:
            latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        if self.sd_version == 'SANA':
            text_embeds, attn_mask = self.get_sana_text_embeds(prompts, negative_prompts) #[2, 300, 2304]
            text_embeds = text_embeds.to(dtype=self.unet.dtype, device=self.device)
        else:
            text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Define panorama grid and get views
        if self.sd_version == 'SANA':
            latent = torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device, dtype=self.unet.dtype
            )        
        else:
            latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        # with torch.autocast('cuda'):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:
                    # TODO we can support batches, and pass multiple views at once to the unet
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    # NOTE : https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/sana_transformer.py
                    if self.sd_version == 'SANA':
                        latent_model_input = torch.cat([latent_view] * 2, dim=0) #[2, 32, 64, 64]
                        batch = latent_model_input.shape[0] # 2
                        if isinstance(t, torch.Tensor):
                            timesteps = t.reshape(-1)
                        else:
                            timesteps = torch.tensor([t])
                            # tensor([999,999])

                        timesteps = timesteps.repeat(batch).view(-1).to(torch.long).to(self.device)
                        out = self.unet(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=text_embeds,
                            timestep=timesteps,
                            encoder_attention_mask=attn_mask,
                            return_dict=True,
                        )
                        noise_pred = out.sample
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # compute the denoising step with the reference model
                        # latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                        value[:, :, h_start:h_end, w_start:w_end] += noise_pred
                        count[:, :, h_start:h_end, w_start:w_end] += 1
                        noise_pred_full = torch.where(count>0,value/count, value)

                    else:
                        latent_model_input = torch.cat([latent_view] * 2)
                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                        # perform guidance
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        # compute the denoising step with the reference model
                        latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                        value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                        count[:, :, h_start:h_end, w_start:w_end] += 1
                    
                if self.sd_version == 'SANA':
                    latent=self.scheduler.step(noise_pred_full, t, latent)['prev_sample']
                else:
                    latent = torch.where(count > 0, value / count, value)
                

        # Img latents -> imgs
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='a photo of the dolomites')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0', '2.1','SANA'],
                        help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--outfile', type=str, default='out.png')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use')
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device(f'cuda:{opt.gpu}')

    sd = MultiDiffusion(device, opt.sd_version)

    img = sd.text2panorama(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # save image
    img.save(opt.outfile)
