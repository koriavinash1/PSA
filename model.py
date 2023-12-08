from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, nn

from einops import rearrange, reduce, repeat

EPS = -9  # minimum logscale
activation = nn.ReLU()

@torch.jit.script
def gaussian_kl(
    q_loc: Tensor, q_logscale: Tensor, p_loc: Tensor, p_logscale: Tensor
) -> Tensor:
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2))
        / p_logscale.exp().pow(2)
    )


@torch.jit.script
def sample_gaussian(loc: Tensor, logscale: Tensor) -> Tensor:
    return loc + logscale.exp() * torch.randn_like(loc)


class Block(nn.Module):
    def __init__(
        self,
        in_width: int,
        bottleneck: int,
        out_width: int,
        kernel_size: int = 3,
        residual: bool = False,
        down_rate: Optional[int] = None,
        upsample_rate: Optional[int] = None,
        version: Optional[str] = None,
    ):
        super().__init__()
        self.d = down_rate
        self.u = upsample_rate
        self.residual = residual
        padding = 0 if kernel_size == 1 else 1

        if version == "vlight":  # uses less VRAM
            activation = nn.ReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(in_width, out_width, kernel_size, 1, padding),
                activation
            )
        elif version == "light":  # uses less VRAM
            activation = nn.ReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(in_width, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, out_width, kernel_size, 1, padding),
                activation,
            )
        else:  # for morphomnist
            activation = nn.GELU()
            self.conv = nn.Sequential(
                nn.Conv2d(in_width, bottleneck, 1, 1),
                activation,
                nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, out_width, 1, 1),
                activation,
            )

        if self.residual and (self.d or in_width > out_width):
            self.width_proj = nn.Conv2d(in_width, out_width, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        
        if self.residual:
            if x.shape[1] != out.shape[1]:
                x = self.width_proj(x)
            out = x + out

        if self.d:
            if isinstance(self.d, float):
                out = F.adaptive_avg_pool2d(out, int(out.shape[-1] / self.d))
            else:
                out = F.avg_pool2d(out, kernel_size=self.d, stride=self.d)
        
        if self.u:
            out = F.interpolate(out, scale_factor=self.u)
        
        return out


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # parse architecture
        stages = []
        activation = nn.ReLU()
        for i, stage in enumerate(args.enc_arch.split(",")):
            start = stage.index("b") + 1
            end = stage.index("d") if "d" in stage else None
            n_blocks = int(stage[start:end])

            if i == 0:  # define network stem
                if n_blocks == 0 and "d" not in stage:
                    print("Using stride=2 conv encoder stem.")
                    stem_width, stem_stride = args.channels[1], 2
                    continue
                else:
                    stem_width, stem_stride = args.channels[0], 1
                self.stem = nn.Sequential(
                    nn.Conv2d(
                            args.input_channels,
                            stem_width,
                            kernel_size=7,
                            stride=stem_stride,
                            padding=3,
                            ),
                    activation
                )


            stages += [(args.channels[i], None) for _ in range(n_blocks)]
            if "d" in stage:  # downsampling block
                stages += [(args.channels[i + 1], int(stage[stage.index("d") + 1]))]
        blocks = []
        for i, (width, d) in enumerate(stages):
            prev_width = stages[max(0, i - 1)][0]
            bottleneck = int(prev_width / args.bottleneck)
            blocks.append(
                Block(prev_width, bottleneck, width, down_rate=d, version=args.vr)
            )
        for b in blocks:
            b.conv[-2].weight.data *= np.sqrt(1 / len(blocks))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            res = x.shape[2]
            if res % 2 and res > 1:  # pad if odd resolution
                x = F.pad(x, [0, 1, 0, 1])
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # parse architecture

        # ===========================================
        stages = []
        for i, stage in enumerate(args.dec_arch.split(",")):
            start = stage.index("b") + 1
            end = stage.index("u") if "u" in stage else None
            n_blocks = int(stage[start:end])
            res = int(stage[:start-1])
            stages += [(args.channels[::-1][i], res, None) for _ in range(n_blocks)]
            
            if "u" in stage:  # downsampling block
                stages += [(args.channels[::-1][i + 1], res, int(stage[stage.index("u") + 1]))]


        self.blocks = []
        for i, (width, res, u) in enumerate(stages):
            prev_width = stages[max(0, i - 1)][0]
            bottleneck = int(prev_width / args.bottleneck)
            self.blocks.append(
                Block(prev_width, bottleneck, width, upsample_rate = u, version=args.vr)
            )
        self.blocks = nn.ModuleList(self.blocks)
       
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        for block in self.blocks:
            x = block(x)
            res = x.shape[2]
            if res % 2 and res > 1:  # pad if odd resolution
                x = F.pad(x, [0, 1, 0, 1])
        return x


class DGaussNet(nn.Module):
    def __init__(self, args):
        super(DGaussNet, self).__init__()
        self.x_loc = nn.Conv2d(
            args.input_channels, args.input_channels, kernel_size=1, stride=1
        )
        self.x_logscale = nn.Conv2d(
            args.input_channels, args.input_channels, kernel_size=1, stride=1
        )

        if args.input_channels == 3:
            self.channel_coeffs = nn.Conv2d(args.input_channels, 3, kernel_size=1, stride=1)

        if args.std_init > 0:  # if std_init=0, random init weights for diag cov
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, np.log(args.std_init))

            covariance = args.x_like.split("_")[0]
            if covariance == "fixed":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = False
            elif covariance == "shared":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = True
            elif covariance == "diag":
                self.x_logscale.weight.requires_grad = True
                self.x_logscale.bias.requires_grad = True
            else:
                NotImplementedError(f"{args.x_like} not implemented.")

    
    def forward(
        self, h: Tensor, x: Optional[Tensor] = None, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=EPS)

        # for RGB inputs
        if hasattr(self, "channel_coeffs"):
            coeff = torch.tanh(self.channel_coeffs(h))
            if x is None:  # inference
                # loc = loc + logscale.exp() * torch.randn_like(loc)  # random sampling
                f = lambda x: torch.clamp(x, min=-1, max=1)
                loc_red = f(loc[:, 0, ...])
                loc_green = f(loc[:, 1, ...] + coeff[:, 0, ...] * loc_red)
                loc_blue = f(
                    loc[:, 2, ...]
                    + coeff[:, 1, ...] * loc_red
                    + coeff[:, 2, ...] * loc_green
                )
            else:  # training
                loc_red = loc[:, 0, ...]
                loc_green = loc[:, 1, ...] + coeff[:, 0, ...] * x[:, 0, ...]
                loc_blue = (
                    loc[:, 2, ...]
                    + coeff[:, 1, ...] * x[:, 0, ...]
                    + coeff[:, 2, ...] * x[:, 1, ...]
                )

            loc = torch.cat(
                [loc_red.unsqueeze(1), loc_green.unsqueeze(1), loc_blue.unsqueeze(1)],
                dim=1,
            )

        if t is not None:
            logscale = logscale + torch.tensor(t).to(h.device).log()
        return loc, logscale

    def approx_cdf(self, x: Tensor) -> Tensor:
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, h: Tensor, x: Tensor) -> Tensor:
        loc, logscale = self.forward(h, x)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(
        self, h: Tensor, return_loc: bool = True, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, logscale.exp()



class GaussSlotConditioningNet(nn.Module):
    def __init__(self, args):
        super(GaussSlotConditioningNet, self).__init__()
        slot_dim = 5
        condition_info = args.compositional_conditioning

        nconditions = 0
        if 'color' in condition_info.lower():
            nconditions += 4 # R, G, B,Mask 
        
        if 'com' in condition_info.lower():
            nconditions += 2


        resolution = (args.input_res, args.input_res)
        xy = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        xx, yy = torch.meshgrid(xy, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        grid = grid.unsqueeze(0)
        
        self.x_loc = nn.Linear(slot_dim, nconditions)
        self.x_logscale = nn.Linear(slot_dim, nconditions)

        if self.learn_cprior:
            self.mu_p_c = nn.Parameter(torch.zeros(1, nconditions), requires_grad=True)
            nn.init.xavier_normal_(self.mu_p_c)

            self.logscale_p_c = nn.Parameter(torch.ones(1, nconditions), requires_grad=True)
            nn.init.xavier_normal_(self.logscale_p_c)
        else:
            self.register_buffer("mu_p_c", torch.zeros(1, nconditions))
            self.register_buffer("logscale_p_c", torch.ones(1, nconditions))

    
    def forward(
        self, h: Tensor, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        # H: B, C, H, W
        # TODO: need further thinking
        # com = 
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=EPS)

        if t is not None:
            logscale = logscale + torch.tensor(t).to(h.device).log()
        return loc, logscale


    def sample(
        self, return_loc: bool = True, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, logscale.exp()


class SoftPositionEmbed(nn.Module):
    def __init__(self, out_channels: int, resolution: Tuple[int, int]):
        super().__init__()
        # (1, height, width, 4)
        self.register_buffer("grid", self.build_grid(resolution))
        self.mlp = nn.Linear(4, out_channels)  # 4 for (x, y, 1-x, 1-y)

    def forward(self, x: Tensor):
        # (1, height, width, out_channels)
        grid = self.mlp(self.grid)
        # (batch_size, out_channels, height, width)
        return x + grid.permute(0, 3, 1, 2)

    def build_grid(self, resolution: Tuple[int, int]) -> Tensor:
        xy = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        xx, yy = torch.meshgrid(xy, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        grid = grid.unsqueeze(0)
        return torch.cat([grid, 1.0 - grid], dim=-1)




class SlotAttentionWithPositions(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.nslots = args.nslots
        self.niters = args.niters
        self.implicit = args.implicit
        self.key_dim = args.channels[-1]
        self.use_gru = (self.niters > 0)

        self.embedding_dim = args.channels[-1]

        self.EM = args.EM_slots.split('_')[0].lower() == 'yes'
        if self.EM: self.EM_type = args.EM_slots.split('_')[1].lower()
        if self.EM: self.EM_map = args.EM_slots.split('_')[2].lower() == 'map'

        self.no_additive_decoder = args.no_additive_decoder

        self.init_spatial_resolution = args.initial_decoder_spatial_resolution

        self.eps = 1e-8
        self.scale = self.key_dim ** -0.5

        activation = nn.ReLU() # nn.GELU()
        # =====================================================
        self.norm_input  = nn.LayerNorm(self.embedding_dim)

        if (not self.EM) or (self.EM_type == 'dynamic'):
            self.to_q = nn.Linear(self.key_dim, self.key_dim)
            self.to_k = nn.Linear(self.embedding_dim, self.key_dim)
            self.to_v = nn.Linear(self.embedding_dim, self.key_dim)

            if self.use_gru:
                self.gru  = nn.GRUCell(self.key_dim, self.key_dim)


            # slot transformation
            self.norm_pre_ff         = nn.LayerNorm(self.key_dim)
            self.slot_transformation = nn.Sequential(nn.Linear(self.key_dim, self.key_dim),
                                                    activation,
                                                    nn.Linear(self.key_dim, self.embedding_dim),
                                                    activation)

            self.norm_slots  = nn.LayerNorm(self.key_dim)


        self.deterministic_slot_init = args.model in ['SSA', 'SSAU']


        if not self.deterministic_slot_init:
            variational_slot_models = ['ASA', 'VASA']

            self.variational_slots = args.model in variational_slot_models
            # Slot sampling stratergy ==============================
            if not self.variational_slots:
                # global priors
                self.slots_loc    = nn.Parameter(torch.randn(1, self.key_dim))
                self.slots_logscale = nn.Parameter(torch.rand(1, self.key_dim))
            else:
                self.stochastic_slots = True
                self.z_to_slots = nn.Sequential(
                                nn.Linear(self.embedding_dim, self.key_dim),
                                activation,
                                nn.Linear(self.key_dim, self.key_dim),
                            )

                self.slots_loc = nn.Sequential(
                                nn.Linear(self.key_dim, self.key_dim),
                                activation,
                                nn.Linear(self.key_dim, self.key_dim),
                            )

                self.slots_logscale = nn.Sequential(
                                nn.Linear(self.key_dim, self.key_dim),
                                activation,
                                nn.Linear(self.key_dim, self.key_dim),
                            )
        else:
            self.init_slot_transformation = nn.Sequential(
                            nn.Linear(self.embedding_dim , self.key_dim),
                            activation,
                            nn.Linear(self.key_dim, self.key_dim),
                        )

      
        nconditions = args.nconditions
        if (nconditions > 0):
            self.conditioning = nn.Linear(self.key_dim + nconditions, self.key_dim)


        # ===========================================
        # encoder postional embedding with linear transformation
        res = int(args.enc_arch.split(',')[-1].split('b')[0])
        ntokens = res**2
        self.encoder_norm        = nn.LayerNorm([ntokens, self.embedding_dim])
        self.encoder_position    = SoftPositionEmbed(self.embedding_dim, (res, res))
        self.encoder_feature_mlp = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                                activation,
                                                nn.Linear(self.embedding_dim, self.embedding_dim),
                                                activation)


        # ============================================
        # decoder setting
        # decoder positional embeddings
        resolution = (self.init_spatial_resolution, self.init_spatial_resolution)
        self.decoder_position    = SoftPositionEmbed(self.key_dim, resolution)

        if self.no_additive_decoder:
            layers = []

            nblocks = 1
            for iconv in range(nblocks):
                block = nn.Sequential(
                                Block(self.key_dim,
                                        self.key_dim,
                                        self.key_dim,
                                        kernel_size = 3,
                                        residual=False,
                                        version=args.vr,
                                    )
                        )
                layers.append(block)


            layers.append(Block(self.key_dim,
                                        self.key_dim,
                                        self.key_dim + 1,
                                        kernel_size = 3,
                                        residual=False,
                                        version=args.vr,
                                    ))
            self.decode = nn.Sequential(*layers)


    def get_stochastic_initial_slots(self, 
                                        features: Optional[Tensor] = None, 
                                        nsamples: int = 1, 
                                        nslots: Optional[int] = None,
                                        device: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        # features: B x channels x res x res
        nsamples = nsamples if features is None else features.shape[0] 
        nslots = nslots if nslots else self.nslots


        if self.variational_slots:
            features = features.flatten(-2, -1).mean(-1) 
            slots = self.z_to_slots(features)
            slots_loc = self.slots_loc(slots)
            slots_logscale = self.slots_logscale(slots)
        else:
            slots_loc = self.slots_loc.expand(nsamples, -1)
            slots_logscale = self.slots_logscale.expand(nsamples, -1)


        slots = slots_loc.unsqueeze(1) +\
                (slots_logscale).unsqueeze(1).exp() * torch.randn(
                    nsamples, nslots, self.key_dim, device=device
                )

        return (slots, slots_loc.unsqueeze(-1).unsqueeze(-1),\
                 slots_logscale.unsqueeze(-1).unsqueeze(-1))



    def generate_aggregate_posterior(self, inputs: Tensor,
                    routing_iters: int = 10,
                    inputs_loc: Optional[Tensor] = None,
                    inputs_logscale: Optional[Tensor] = None, 
                    properties: Optional[Tensor] = None,
                    num_slots: Optional[Tensor] = None) -> dist.MixtureSameFamily:
        # x: b, c, h
        
        b, d, h, w = inputs.shape
        
        # =======================
        # initialise slots 
        if not self.deterministic_slot_init:
            slots, init_slot_loc, init_slot_logscale = self.get_stochastic_initial_slots(inputs, 
                                                                    nslots = num_slots, 
                                                                    device= inputs.device)
            if self.variational_slots:
                slot_loss = gaussian_kl(init_slot_loc, 
                                    init_slot_logscale, 
                                    torch.zeros_like(init_slot_loc), 
                                    torch.ones_like(init_slot_logscale))
            else:
                slot_loss = torch.zeros_like(inputs)
        else:
            slots = self.get_deterministic_initial_slots(inputs_loc, inputs_logscale, num_slots)
            slot_loss = torch.zeros_like(inputs)


        _, num_slots, _ = slots.shape
        if hasattr(self, 'conditioning'):
            slots = self.conditioning(torch.cat([slots, properties[:, :num_slots, :].to(slots.device)], dim = 2))


        # ========================
        tokens = self.encoder_transformation(inputs)
        tokens = self.norm_input(tokens)        


        if self.EM_type == 'dynamic':
            k, v = self.to_k(tokens), self.to_v(tokens)


        initial_slots = slots.clone()

        pi = torch.ones(b, num_slots, 1, device = slots.device, dtype = slots.dtype)/num_slots
        sigma = init_slot_logscale.squeeze().unsqueeze(1).repeat(1, num_slots, 1)
        sigma = sigma.exp() + self.eps

        sigma_initial = torch.var(tokens, dim=1, keepdim=True) + self.eps # B, 1, d
        sigma_initial /= (num_slots**(1/d))

        attn = torch.ones(tokens.shape[0], num_slots, tokens.shape[1], device = slots.device, dtype = slots.dtype)

        for _ in range(routing_iters):
            if self.EM_type == 'dynamic':
                slots, pi, sigma, attn = self.EM_step(slots, sigma, sigma_initial, pi, k, v)
            else:
                slots, pi, sigma, attn = self.EM_fixed_step(slots, sigma, sigma_initial, pi, tokens)



        if self.implicit: 
            if self.EM_type == 'dynamic':
                slots, pi, sigma, attn = self.EM_step(slots, sigma, pi, k, v)
            else:
                slots, pi, sigma, attn = self.EM_fixed_step(slots, sigma, pi, tokens)


        joint_pi = pi.flatten(0, 1)/slots.shape[0]
        joint_sigma = sigma.flatten(0, 1) # B*K, d
        joint_slots = slots.flatten(0, 1) # B*K, d
        
        mix = dist.Categorical(joint_pi.squeeze(1))
        comp = dist.Independent(dist.Normal(joint_slots, joint_sigma), 1)

        self.aggregate_posterior = dist.MixtureSameFamily(mix, comp)
        return self.aggregate_posterior



    def get_deterministic_initial_slots(self, 
                                            features_loc: Tensor, 
                                            features_logscale: Tensor,
                                            nslots: Optional[int] = None) -> Tensor:
        # features_loc: B x channels x res x res
        nslots = nslots if nslots else self.nslots

        features_loc = features_loc.flatten(-2, -1).permute(0, 2, 1)
        features_logscale = features_logscale.flatten(-2, -1).permute(0, 2, 1)

        b, n, c = features_loc.shape
        xslot = (features_loc.unsqueeze(1) + \
                    (features_logscale).exp().unsqueeze(1) * torch.randn(
                        b, nslots, n, c, device=features_loc.device)).mean(2)

        return self.init_slot_transformation(xslot)
    

    def encoder_transformation(self, features: Tensor) -> Tensor:
        #features: B x W x H x C
        features = self.encoder_position(features).permute(0, 2, 3, 1)
        features = torch.flatten(features, 1, 2)
        features = self.encoder_norm(features)
        features = self.encoder_feature_mlp(features)
        return features

    
    def slotfeature_competition(self, b: int, x: Tensor) -> Tensor:
        bk, c, h, w = x.shape

        x = x.view(b, -1, c, h, w)
        recons, masks = torch.split(x, [c-1, 1], dim=2)
        masks = masks.softmax(dim=1)

        # (b, c, h, w)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks


    def decoder_transformation(self, slots: Tensor, pi: Optional[Tensor] = None) -> Tensor:
        # features: B x nslots x dim
        B, K, d = slots.shape

        if pi is None:
            pi = torch.ones_like(slots)/K

        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(2).unsqueeze(3) # (B*nslots) x dim
        features = slots.repeat((1, 1, self.init_spatial_resolution, self.init_spatial_resolution))
        features = self.decoder_position(features)

        if self.no_additive_decoder:
            features = self.decode(features)
            features, _, _ = self.slotfeature_competition(B, features)

            # features = features.view(B, K, d, self.init_spatial_resolution, self.init_spatial_resolution)
            # features = features * pi.unsqueeze(-1).unsqueeze(-1)

            # features = features.sum(dim = 1)

        return features



    def step(self, slots_prev: Tensor, 
                    k: Tensor, 
                    v: Tensor) -> Tuple[Tensor, Tensor]:

        slots_prev= self.norm_slots(slots_prev)
        q = self.to_q(slots_prev)

        dots = torch.einsum('bid, bjd -> bij', q, k) * self.scale
        attn_vis = dots.softmax(dim=1) + self.eps
        attn = attn_vis / attn_vis.sum(dim=-1, keepdim=True)

        slots = torch.einsum('bjd,bij->bid', v, attn)

        if self.use_gru:
            slots = self.gru(
                rearrange(slots, 'b n d -> (b n) d'),
                rearrange(slots_prev, 'b n d -> (b n) d')
            )

            slots = slots.reshape(-1, self.nslots, self.key_dim)

        slots = slots + self.slot_transformation(self.norm_pre_ff(slots))

        return slots, attn_vis


    def EM_step(self, slots_prev: Tensor,
                    sigma_prev :Tensor, 
                    sigma_initial: Tensor,
                    pi_prev: Tensor,
                    k: Tensor, 
                    v: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # Sigma: K x d
        # K: N x d
        # pi_prev: K x 1

        b , N, d = k.shape
        _, K, _ = slots_prev.shape

        slots_prev= self.norm_slots(slots_prev)
        q_transform = self.to_q(slots_prev)


        # E-step
        log_pi    = - 0.5 * torch.tensor(2 * torch.pi, device=k.device).log()
        log_scale = - torch.log(torch.clamp(sigma_prev.unsqueeze(2), min = self.eps)) # (B, K, 1, d)
        exponent  = - 0.5 * (k.unsqueeze(1) - q_transform.unsqueeze(2)) ** 2 / (sigma_prev.unsqueeze(2)) ** 2 # (B, K, N, d)
        log_probs = torch.log(torch.clamp(pi_prev, min = self.eps)) + (exponent + log_pi + log_scale).sum(dim=-1) # (B, K, N)
                         
        attn = log_probs.softmax(dim=1) + self.eps # (B, K, N)


        # M-step
        Nk = torch.sum(attn, dim=2, keepdim=True) # (B, K, 1)
        pi = Nk / N
        
        slots = (1 / Nk) * torch.sum(attn.unsqueeze(-1) * v.unsqueeze(1), dim=2) # (B, K, D)

        
        if not self.EM_map:
            sigma = (1 / Nk) * torch.sum(attn.unsqueeze(-1) * (k.unsqueeze(1) - \
                                                        self.to_q(slots).unsqueeze(2))**2, dim=2) # (B, K, D)
        else:
            sigma = (self.to_q(sigma_initial) + torch.sum(attn.unsqueeze(-1) * (k.unsqueeze(1) - \
                                        self.to_q(slots_prev.unsqueeze(2))) ** 2, dim=2)) / (Nk + 2*d + 4)  # (B, K, D)
    
            
        sigma = torch.sqrt(sigma) + self.eps

        if self.use_gru:   
            slots = self.gru(
                rearrange(slots, 'b n d -> (b n) d'),
                rearrange(slots_prev, 'b n d -> (b n) d')
            )

            slots = slots.reshape(-1, self.nslots, self.key_dim)

        slots = slots + self.slot_transformation(self.norm_pre_ff(slots))

        return slots, pi, sigma, attn


    def EM_fixed_step(self, slots_prev: Tensor,
                    sigma_prev :Tensor, 
                    sigma_initial: Tensor,
                    pi_prev: Tensor,
                    z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # Sigma: K x d
        # K: N x d
        # pi_prev: K x 1

        b , N, d = z.shape
        _, K, _ = slots_prev.shape


        # E-step
        log_pi    = - 0.5 * torch.tensor(2 * torch.pi, device=z.device).log()
        log_scale = - torch.log(torch.clamp(sigma_prev.unsqueeze(2), min = self.eps)) # (B, K, N, d)
        exponent  = - 0.5 * (z.unsqueeze(1) - slots_prev.unsqueeze(2)) ** 2 / (sigma_prev.unsqueeze(2)) ** 2 # (B, K, N, d)
        log_probs = torch.log(torch.clamp(pi_prev, min = self.eps)) + (exponent + log_pi + log_scale).sum(dim=-1) # (B, K, N)
                         
        attn = log_probs.softmax(dim=1) + self.eps # (B, K, N)


        # M-step
        Nk = torch.sum(attn, dim=2, keepdim=True) # (B, K, 1)
        pi = Nk / N
        
        slots = (1 / Nk) * torch.sum(attn.unsqueeze(-1) * z.unsqueeze(1), dim=2) # (B, K, D)

        if not self.EM_map:
            sigma = (1 / Nk) * torch.sum(attn.unsqueeze(-1) * (z.unsqueeze(1) - \
                                                        slots.unsqueeze(2))**2, dim=2) # (B, K, D)
        else:
            sigma = (sigma_initial + torch.sum(attn.unsqueeze(-1) * (z.unsqueeze(1) - \
                                        slots_prev.unsqueeze(2)) ** 2, dim=2)) / (Nk + 2*d + 4)  # (B, K, D)
    
        sigma = torch.sqrt(sigma) + self.eps

        return slots, pi, sigma, attn


    def forward(self, 
                    inputs: Tensor,
                    inputs_loc: Optional[Tensor] = None,
                    inputs_logscale: Optional[Tensor] = None, 
                    properties: Optional[Tensor] = None,
                    num_slots: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:

        # inputs: b x dim x h x w
        # prev_attns: b x k x ntokens

        b, d, h, w = inputs.shape
        
        # =======================
        # initialise slots 
        if not self.deterministic_slot_init:
            slots, init_slot_loc, init_slot_logscale = self.get_stochastic_initial_slots(inputs, 
                                                                    nslots = num_slots, 
                                                                    device= inputs.device)
            if self.variational_slots:
                slot_loss = gaussian_kl(init_slot_loc, 
                                    init_slot_logscale, 
                                    torch.zeros_like(init_slot_loc), 
                                    torch.ones_like(init_slot_logscale))
            else:
                slot_loss = torch.zeros_like(inputs)
        else:
            slots = self.get_deterministic_initial_slots(inputs_loc, inputs_logscale, num_slots)
            slot_loss = torch.zeros_like(inputs)


        _, num_slots, _ = slots.shape
        if hasattr(self, 'conditioning'):
            slots = self.conditioning(torch.cat([slots, properties[:, :num_slots, :].to(slots.device)], dim = 2))


        # ========================
        tokens = self.encoder_transformation(inputs)
        tokens = self.norm_input(tokens)        


        if (not self.EM) or (self.EM_type == 'dynamic'):
            k, v = self.to_k(tokens), self.to_v(tokens)


        initial_slots = slots.clone()

        pi = torch.ones(b, num_slots, 1, device = slots.device, dtype = slots.dtype)/num_slots

        if self.EM:
            sigma = init_slot_logscale.squeeze().unsqueeze(1).repeat(1, num_slots, 1)
            sigma = sigma.exp() + self.eps

            sigma_initial = torch.var(tokens, dim=1, keepdim=True) + self.eps # B, 1, d
            sigma_initial /= (num_slots**(1/d))

        attn = torch.ones(tokens.shape[0], num_slots, tokens.shape[1], device = slots.device, dtype = slots.dtype)
        for _ in range(self.niters):
            if not self.EM:
                slots, attn = self.step(slots, k, v)    
            else:
                if self.EM_type == 'dynamic':
                    slots, pi, sigma, attn = self.EM_step(slots, sigma, sigma_initial, pi, k, v)
                else:
                    slots, pi, sigma, attn = self.EM_fixed_step(slots, sigma, sigma_initial, pi, tokens)



        if self.implicit: 
            if not self.EM:
                slots, attn = self.step(slots.detach(), k, v)    
            else:
                if self.EM_type == 'dynamic':
                    slots, pi, sigma, attn = self.EM_step(slots, sigma, pi, k, v)
                else:
                    slots, pi, sigma, attn = self.EM_fixed_step(slots, sigma, pi, tokens)


        # print (sigma.max(), sigma.min(), sigma.mean(), torch.var(attn, 1).mean())

        # if self.EM:     
        #     slots = slots + sigma * torch.randn_like(slots)


        return self.decoder_transformation(slots, pi), attn, slot_loss, (initial_slots, slots)


class SlotAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.vr = "vlight" # if "ukbb" in args.hps else None  # hacky
        self.encoder = Encoder(args)

        variational_models = ['VAE', 'VSA', 'VASA', 'SSA', 'SSAU']
        bottleneck = args.channels[-1]//args.bottleneck

        self.variational_latents = args.model in variational_models

        if self.variational_latents:    
            
            self.zprior_type = args.zprior.lower()
            self.learn_prior  = args.learn_prior  

            self.latent_dim = args.channels[-1]
            self.res = res = int(args.enc_arch.split(',')[-1].split('b')[0])


            self.posterior = Block(
                            args.channels[-1],
                            bottleneck,
                            2 * args.channels[-1],
                            residual=False,
                            version=args.vr,
                        )

            # self.posterior = nn.Sequential(
            #                 nn.Linear(self.latent_dim*self.res*self.res, self.latent_dim*self.res*self.res),
            #                 nn.GELU(),
            #                 nn.Linear(self.latent_dim*self.res*self.res, 2*self.latent_dim*self.res*self.res),
            #                 )


            if args.zprior == 'gmm':
                self.num_components = args.num_components
                self.pi_p_c = nn.Parameter(torch.ones(self.num_components) / self.num_components,
                                        requires_grad=self.learn_prior)
                self.mu_p_z = nn.Parameter(torch.zeros(self.num_components, self.latent_dim), 
                                            requires_grad=self.learn_prior)
                nn.init.xavier_normal_(self.mu_p_z)
                self.logscale_p_z = nn.Parameter(torch.Tensor(self.num_components, 
                                            self.latent_dim), requires_grad=self.learn_prior)
                nn.init.xavier_normal_(self.logscale_p_z)
                
            elif (args.zprior == 'gauss'):
                
                if self.learn_prior:
                    self.mu_p_z = nn.Parameter(torch.zeros(1, self.latent_dim, res, res), 
                                                        requires_grad=True)
                    nn.init.xavier_normal_(self.mu_p_z)

                    self.logscale_p_z = nn.Parameter(torch.ones(1, self.latent_dim, res, res), 
                                                        requires_grad=True)
                    nn.init.xavier_normal_(self.logscale_p_z)
                else:
                    self.register_buffer("mu_p_z", torch.zeros(1, self.latent_dim, res, res))
                    self.register_buffer("logscale_p_z", torch.ones(1, self.latent_dim, res, res))

        

        # ========================================================
        self.model = args.model.lower() 
        if args.model.lower() == 'vae':
            self.no_additive_decoder = args.no_additive_decoder = True
            eres = int(args.enc_arch.split(',')[-1].split('b')[0])
            dblock = args.dec_arch.split(',')[0]

            dres = int(dblock.split('b')[0])
            usample = int(dblock.split('u')[-1])
            ndblock = dblock.replace(f'{dres}b', f'{eres}b')
            ndblock = ndblock.replace(f'u{usample}', f'u{int(dres*usample/eres)}')
            
            args.dec_arch = args.dec_arch.replace(dblock, ndblock)

        else:

            self.num_slots = args.nslots
            self.no_additive_decoder = args.no_additive_decoder
            self.slot_attention = SlotAttentionWithPositions(args)
    

        self.decoder = Decoder(args)

        # ========================================================


        # Heirarchy setup
        # self.eres = int(args.enc_arch.split(',')[-1].split('b')[0])
        # if self.variational_latents and self.eres > 0:
        #     self.heirarchy_projection = Block(
        #                             args.channels[-1],
        #                             bottleneck,
        #                             2 * args.channels[-1],
        #                             residual=False,
        #                             version=args.vr,
        #                         )
                    
        #     # 1x1 setup
        #     if self.learn_prior:
        #         self.mu_p_z1x1 = nn.Parameter(torch.zeros(1, self.latent_dim, 1, 1), 
        #                                             requires_grad=True)
        #         nn.init.xavier_normal_(self.mu_p_z1x1)

        #         self.logscale_p_z1x1 = nn.Parameter(torch.ones(1, self.latent_dim, 1, 1), 
        #                                             requires_grad=True)
        #         nn.init.xavier_normal_(self.logscale_p_z1x1)
        #     else:
        #         self.register_buffer("mu_p_z1x1", torch.zeros(1, self.latent_dim, 1, 1))
        #         self.register_buffer("logscale_p_z1x1", torch.ones(1, self.latent_dim, 1, 1))
           


        if not self.no_additive_decoder:
            self.projection = nn.Conv2d(
                            args.channels[0], 
                            args.input_channels + 1, 
                            kernel_size=1, stride=1
                        )
        else:
            self.projection = nn.Conv2d(
                            args.channels[0], 
                            args.input_channels, 
                            kernel_size=1, stride=1
                        )




        if args.x_like == 'mse':
            self.likelihood = nn.MSELoss()
            self.x_like = 'mse'
        elif args.x_like.split("_")[1] == "dgauss":
            self.likelihood = DGaussNet(args)
            self.x_like = 'dgauss'
        else:
            NotImplementedError(f"{args.x_like} not implemented.")
        
        self.free_bits = args.kl_free_bits
        self.register_buffer("log2", torch.tensor(2.0).log())



    def _mixture_kl_loss(self, z: Tensor, loc: Tensor, logscale: Tensor):
        # loc, scale: B x ds
        # slots: B x K x ds

        z = z.permute(0, 2, 3, 1).flatten(1, 2)
        loc = loc.permute(0, 2, 3, 1).flatten(1, 2)
        logscale = logscale.permute(0, 2, 3, 1).flatten(1, 2)

        b, k, c = z.shape

        z = z.flatten(0, 1) #treat each slot independent from others
        loc = loc.flatten(0, 1)
        logscale = logscale.flatten(0, 1)

        scale = torch.exp(logscale)
        q_z = dist.Normal(loc, scale)

        std_pz_c = torch.exp(self.logscale_p_z)
        pz_c = dist.Independent(dist.Normal(loc=self.mu_p_z, scale=std_pz_c), 1)

        log_prob_pz_c = pz_c.log_prob(z.unsqueeze(-2))
        log_prob_p_c = torch.log_softmax(self.pi_p_c, dim=-1)
        q_c_x = torch.softmax(log_prob_pz_c + log_prob_p_c, dim=1)


        # compute the MC estimate of E_{q(z,c|x)}[log q(z|x)] where z~q(z|x)
        log_prob_E_qz_x = q_z.log_prob(z).mean(1)
        
        # compute the MC estimate of E_{q(z,c|x)}[log p(z|c)] where z~q(z|x)
        log_prob_E_pz_c = torch.sum(q_c_x * log_prob_pz_c, dim=1)    
        
        # compute E_{q(z,c|x)}[log q(c|x)]
        log_prob_E_q_c_x = torch.sum(q_c_x * torch.log(torch.clamp(q_c_x, min=1e-6)), dim=1)

        # compute E_{q(z,c|x)}[log p(c)]
        log_prob_E_p_c = torch.sum(q_c_x * torch.log(self.pi_p_c).unsqueeze(0), dim=1)

        KL_z = (log_prob_E_qz_x - log_prob_E_pz_c).view(b, k, 1, 1)
        KL_c = (log_prob_E_q_c_x - log_prob_E_p_c).view(b, k, 1, 1)

        KL = KL_z + KL_c 
   
        return KL 
        

    
    def get_latents(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b, c, hs, ws = h.shape

        kl_loss1x1 = 0

        # heirarchy projection
        # if self.eres > 1:
        #     h1x1 = F.adaptive_avg_pool2d(h, output_size=(1, 1))
        #     loc, logscale = self.heirarchy_projection(h1x1).chunk(2, dim=1)
        #     z = loc + (logscale).exp()*torch.randn_like(loc)

        #     kl_loss1x1 = gaussian_kl(loc, logscale, 
        #                             self.mu_p_z1x1, 
        #                             self.logscale_p_z1x1)
        #     z = F.interpolate(z, size=(hs, ws))
        #     h = z


        # h = h.view(b, -1)
        loc, logscale = self.posterior(h).chunk(2, dim=1)
        
        # loc = loc.view(b, c, hs, ws)
        # logscale = logscale.view(b, c, hs, ws)
        
        z = loc + (logscale).exp()*torch.randn_like(loc)


        if self.zprior_type == 'gmm':
            kl_loss = self._mixture_kl_loss(z, loc, logscale)
        else:
            kl_loss = gaussian_kl(loc, logscale, 
                                    self.mu_p_z, 
                                    self.logscale_p_z)
        
        return z, loc, logscale, kl_loss, kl_loss1x1



    def slotpixel_competition(self, b: int, h: Tensor) -> Tensor:

        x = self.projection(h)
        bk, c, h, w = x.shape

        x = x.view(b, -1, c, h, w)
        recons, masks = torch.split(x, [c-1, 1], dim=2)
        masks = masks.softmax(dim=1)

        # (b, c, h, w)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks



    def forward(self, x: Tensor, 
                    properties: Optional[Tensor] = None, 
                    num_slots: Optional[int] = None, 
                    beta: int = 1) -> Dict[str, Tensor]:
        h = self.encoder(x)
        
        stats = []
        h_loc, h_logscale = None, None 
        if hasattr(self, 'posterior'):
            h, h_loc, h_logscale, kl_loss, kl_loss1x1 = self.get_latents(h)
            stats.append(dict(kl=kl_loss))

            if not isinstance(kl_loss1x1, int):
                stats.append(dict(kl=kl_loss1x1))

        if not (self.model == 'vae'):
            zs, attn, slot_kl_loss, _ = self.slot_attention(h, h_loc, h_logscale, properties, num_slots)
            stats.append(dict(kl=slot_kl_loss))
        else:
            zs = h


        xh = self.decoder(zs)
        if not self.no_additive_decoder:
            xh, recons, masks = self.slotpixel_competition(x.shape[0], xh)
        else:
            xh = self.projection(xh)


        
        if self.x_like == 'mse':
            nll_pp = self.likelihood(xh, x)
        else:
            nll_pp = self.likelihood.nll(xh, x)


        if self.free_bits > 0:
            free_bits = torch.tensor(self.free_bits).type_as(nll_pp)
            kl_pp = torch.zeros(1, dtype=x.dtype, device=x.device).mean()
            for stat in stats:
                kl_pp += torch.maximum(
                    free_bits, stat["kl"].sum(dim=(2, 3)).mean(dim=0)
                ).sum()
        else:
            kl_pp = torch.zeros(1, dtype=x.dtype, device=x.device).mean()
            for _, stat in enumerate(stats):
                kl_pp_ = stat["kl"].sum(dim=(1, 2, 3))
                kl_pp_ = kl_pp_ / np.prod(x.shape[1:])  # per pixel
                kl_pp += kl_pp_.mean()  # / self.log2

        nll_pp = nll_pp.mean()  # / self.log2
        nelbo = nll_pp + beta * kl_pp  # negative elbo (free energy)
        return dict(elbo=nelbo, nll=nll_pp, kl=kl_pp)


    @torch.no_grad()
    def sample(self, 
                nsamples: int = 10, 
                return_loc: bool = True,
                device: int = 0,
                num_slots: Optional[int] = None,
                properties: Optional[Tensor] = None,
                per_component: Optional[bool] = False, 
                t: Optional[float] = None
                ) -> Tuple[Tensor, Tensor]:

        # only in when self.variational_latents is True

        def __vae_latents__():
            z_dist = dist.Normal(loc = self.mu_p_z, 
                                    scale = torch.exp(self.logscale_p_z))
            z_samples = z_dist.sample((nsamples,)).to(device).squeeze(1)

            return z_samples

        if not hasattr(self.slot_attention, 'aggregate_posterior'):
            if not (self.zprior_type == 'gmm'):
                h = __vae_latents__()
            else:
                if self.num_components == 1:
                    h = __vae_latents__()
                
                if not per_component:
                    l_p_c = torch.log_softmax(self.pi_p_c, dim=-1)
                    p_c = dist.one_hot_categorical.OneHotCategorical(logits=l_p_c)
                    c_samples = p_c.sample((nsamples * self.res * self.res, )).to(device)

                    loc = c_samples @ self.mu_p_z
                    scale = c_samples @ torch.exp(self.logscale_p_z)
                    pz_c = dist.Normal(loc=loc, scale=scale)
                    h = pz_c.sample().to(device)
                    h = h.view(nsamples, self.latent_dim, self.res, self.res)
                else:
                    c_samples = torch.tensor(np.eye(self.num_components, dtype=np.float32)).to(device)
                    loc = c_samples @ self.mu_p_z
                    scale = c_samples @ torch.exp(self.logscale_p_z)
                    pz_c = dist.Normal(loc=loc, scale=scale)
                    h = pz_c.sample((nsamples * self.res * self.res,))
                    h = h.view(nsamples, self.latent_dim, self.res, self.res)

            h_loc = self.mu_p_z.repeat(nsamples, 1, 1, 1)
            h_logscale = torch.exp(self.logscale_p_z).repeat(nsamples, 1, 1, 1) 

            if not (self.model == 'vae'):
                zs, attn_maps, _, slots = self.slot_attention(h, h_loc, h_logscale, properties, num_slots)
            else:
                attn_maps = None
                slots = None
                zs = h

        else:
            num_slots = self.num_slots if num_slots is None else num_slots 
            slots = self.slot_attention.aggregate_posterior.sample([num_slots*nsamples])
            pi = self.slot_attention.aggregate_posterior.log_prob(slots)

            slots = slots.view(nsamples, num_slots, -1)
            pi = pi.view(nsamples, num_slots, -1)

            zs = self.slot_attention.decoder_transformation(slots, pi = pi)
            attn_maps = None
            slots = (None, slots)

        xh = self.decoder(zs)
        recons = None; masks = None
        if not self.no_additive_decoder:
            xh, recons, masks = self.slotpixel_competition(nsamples, xh)
        else:
            xh = self.projection(xh)


        if not isinstance(self.likelihood, DGaussNet):
            return (xh, None), recons, masks, attn_maps, slots

        return self.likelihood.sample(xh, return_loc, t=t), recons, masks, attn_maps, slots



    @torch.no_grad()
    def decode_slots(
        self, slots: Tensor,
        t: Optional[float] = None
    ) -> Tensor:

        zs = self.slot_attention.decoder_transformation(slots)
        xh = self.decoder(zs)
        
        if not self.no_additive_decoder:
            xh, recons, masks = self.slotpixel_competition(bs, xh)
        else:
            xh = self.projection(xh)

        if not isinstance(self.likelihood, DGaussNet):
            return (xh, _)

        return self.likelihood.sample(xh, t=t)[0]



    @torch.no_grad()
    def generate_slot_aggregate_posterior(
                    self, latents: Tensor,
                    routing_iters: Optional[int] = 10, 
                    properties: Optional[Tensor] = None,
                    num_slots: Optional[Tensor] = None
            ):

        h_loc, h_logscale = None, None 
        if hasattr(self, 'posterior'):
            latents, h_loc, h_logscale, _, _ = self.get_latents(latents)

        self.slot_attention.generate_aggregate_posterior(latents,
                                                            routing_iters,
                                                            h_loc,
                                                            h_logscale, 
                                                            properties,
                                                            num_slots)


    @torch.no_grad()
    def forward_latents(
        self, latents: Tensor, 
        properties: Optional[Tensor] = None,
        num_slots: Optional[Tensor] = None,
        t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        bs = latents.shape[0]
        h_loc, h_logscale = None, None 
        if hasattr(self, 'posterior'):
            latents, h_loc, h_logscale, _, _ = self.get_latents(latents)


        if not (self.model == 'vae'):
            zs, attn_maps, _, slots = self.slot_attention(latents, h_loc, h_logscale, properties, num_slots)
        else:
            attn_maps = None
            slots = None
            zs = latents

        xh = self.decoder(zs)
        recons = None; masks = None
        if not self.no_additive_decoder:
            xh, recons, masks = self.slotpixel_competition(bs, xh)
        else:
            xh = self.projection(xh)

        if not isinstance(self.likelihood, DGaussNet):
            return (xh,_), recons, masks, attn_maps, slots

        return self.likelihood.sample(xh, t=t), recons, masks, attn_maps, slots