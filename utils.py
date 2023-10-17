import copy
import os
import random
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn
import torch.nn.functional as F

from hps import Hparams


def seed_all(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f


def beta_anneal(beta, step, anneal_steps):
    return min(beta, (max(1e-11, step) / anneal_steps) ** 2)


def normalize(x, x_min=None, x_max=None, zero_one=False):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    print(f"max: {x_max}, min: {x_min}")
    x = (x - x_min) / (x_max - x_min)  # [0,1]
    return x if zero_one else 2 * x - 1  # else [-1,1]


def log_standardize(x):
    log_x = torch.log(x.clamp(min=1e-12))
    return (log_x - log_x.mean()) / log_x.std().clamp(min=1e-12)  # mean=0, std=1


def exists(val):
    return val is not None


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value


class EMA(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = copy.deepcopy(model)
        except:
            print(
                "Your model was not copyable. Please make sure you are not using any LazyLinear"
            )
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


def write_images(args: Hparams, model: nn.Module, batch: Dict[str, Tensor]):
    bs, c, h, w = batch["x"].shape

    if 'properties' not in batch.keys():
        batch['properties'] = None

    # original imgs, channels last, [0,255]
    orig = (batch["x"].permute(0, 2, 3, 1) + 1.0) * 127.5
    device = orig.device
    # properties = batch['properties']
    orig = orig.detach().cpu().numpy().astype(np.uint8)
    viz_images = [orig]

    def postprocess(x: Tensor):
        x = (x.permute(0, 2, 3, 1) + 1.0) * 127.5  # channels last, [0,255]
        return x.detach().cpu().numpy()

    normalize = lambda x: ((x - x.min())/(x.max() - x.min() + 1e-3))*255


    # reconstructions, first abduct z from q(z|x,pa)
    zs = model.encoder(x=batch["x"])

    (x_rec, _), recons, masks, attn = model.forward_latents(latents = zs, 
                                                properties = batch['properties'])
    x_rec = postprocess(x_rec)
    viz_images.append(x_rec.astype(np.uint8))
    viz_images.append(orig * 0)


    _, kslots, ntokens = attn.shape 

    # slot_recons
    # for ik in range(kslots):
    #     recon_slot = recons[:, ik, ...]
    #     recon_slot, _ = model.likelihood(recon_slot) 
    #     slots = recon_slot * masks[:, ik, ...]
    #     slots = slots.permute(0, 2, 3, 1)
    #     viz_images.append(normalize(slots.detach().cpu().numpy()).astype(np.uint8))

    # viz_images.append(orig * 0)


    for ik in range(kslots):
        recon_slot = recons[:, ik, ...]
        recon_slot, _ = model.likelihood(recon_slot)
        slots = recon_slot * masks[:, ik, ...] + (1 - masks[:, ik, ...])
        slots = slots.permute(0, 2, 3, 1)
        viz_images.append(normalize(slots.detach().cpu().numpy()).astype(np.uint8))

    viz_images.append(orig * 0)


    # plot slots at each resolution
    attn = attn * attn.sum(-1, keepdim = True)
    attn = attn.view(bs, kslots, 
                        int(ntokens**0.5), 
                        int(ntokens**0.5))
    _,_, h_enc, w_enc = attn.shape

    attn = attn.repeat_interleave(h // h_enc, dim=-2).repeat_interleave(w // w_enc, dim=-1)
    attn = attn.detach().cpu().numpy()
    
    # for ik in range(kslots):
    #     attn_viz = attn[:, ik, :, :][:, :, :, None]
    #     attn_viz = attn_viz*orig
    #     attn_viz = normalize(attn_viz)

    #     viz_images.append((attn_viz).astype(np.uint8))
        
    # viz_images.append(orig * 0)

    for ik in range(kslots):
        attn_viz = attn[:, ik, :, :][:, :, :, None]
        attn_viz = attn_viz*orig + (1.0 - attn_viz)
        attn_viz = normalize(attn_viz)

        viz_images.append((attn_viz).astype(np.uint8))
        
    viz_images.append(orig * 0)



    if args.model in ['VSA', 'VASA', 'SSA', 'SSAU']:
        nsamples = orig.shape[0]
        for temp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            (x_rec, _), recons, masks, attn = model.sample(nsamples, 
                                                            device = device, 
                                                            properties = batch['properties'], 
                                                            return_loc=True, t=temp)
            x_rec = postprocess(x_rec)

            _, kslots, ntokens = attn.shape 

            # slot_recons
            # for ik in range(kslots):
            #     slots = recons[:, ik, ...] * masks[:, ik, ...]
            #     slots = slots.permute(0, 2, 3, 1)
            #     viz_images.append(normalize(slots.detach().cpu().numpy()).astype(np.uint8))

            # viz_images.append(orig * 0)


            for ik in range(kslots):
                slots = recons[:, ik, ...] * masks[:, ik, ...] + (1 - masks[:, ik, ...])
                slots = slots.permute(0, 2, 3, 1)
                viz_images.append(normalize(slots.detach().cpu().numpy()).astype(np.uint8))

            viz_images.append(orig * 0)

            viz_images.append(x_rec.astype(np.uint8))
            viz_images.append(orig * 0)

    # zero pad each row to have same number of columns for plotting
    for j, img in enumerate(viz_images):
        s = img.shape[0]
        if s < bs:
            pad = np.zeros((bs - s, *img.shape[1:])).astype(np.uint8)
            viz_images[j] = np.concatenate([img, pad], axis=0)


    # concat all images and save to disk
    n_rows = len(viz_images)
    im = (
        np.concatenate(viz_images, axis=0)
        .reshape((n_rows, bs, h, w, c))
        .transpose([0, 2, 1, 3, 4])
        .reshape([n_rows * h, bs * w, c])
    )
    imageio.imwrite(os.path.join(args.save_dir, f"viz-{args.iter}.png"), im)