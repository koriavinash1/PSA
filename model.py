from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, nn

from einops import rearrange, reduce, repeat

EPS = -9  # minimum logscale


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
        residual: bool = True,
        down_rate: Optional[int] = None,
        upsample_rate: Optional[int] = None,
        version: Optional[str] = None,
    ):
        super().__init__()
        self.d = down_rate
        self.u = upsample_rate
        self.residual = residual
        padding = 0 if kernel_size == 1 else 1

        if version == "light":  # uses less VRAM
            activation = nn.ReLU()
            self.conv = nn.Sequential(
                activation,
                nn.Conv2d(in_width, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, out_width, kernel_size, 1, padding),
            )
        else:  # for morphomnist
            activation = nn.GELU()
            self.conv = nn.Sequential(
                activation,
                nn.Conv2d(in_width, bottleneck, 1, 1),
                activation,
                nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, out_width, 1, 1),
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
                self.stem = nn.Conv2d(
                    args.input_channels,
                    stem_width,
                    kernel_size=7,
                    stride=stem_stride,
                    padding=3,
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
            b.conv[-1].weight.data *= np.sqrt(1 / len(blocks))
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
        self.key_dim = args.slot_dim
        self.use_routing = (self.niters > 0)

        self.embedding_dim = args.channels[-1]

        
        self.init_spatial_resolution = args.initial_decoder_spatial_resolution

        self.eps = 1e-8
        self.scale = self.key_dim ** -0.5


        # =====================================================
        self.to_q = nn.Linear(self.key_dim, self.key_dim)
        self.to_k = nn.Linear(self.embedding_dim, self.key_dim)
        self.to_v = nn.Linear(self.embedding_dim, self.key_dim)

        if self.use_routing:
            self.gru  = nn.GRUCell(self.key_dim, self.key_dim)


        # slot transformation
        self.norm_pre_ff         = nn.LayerNorm(self.key_dim)
        self.slot_transformation = nn.Sequential(nn.Linear(self.key_dim, self.key_dim),
                                                nn.GELU(),
                                                nn.Linear(self.key_dim, self.embedding_dim),
                                                nn.GELU())

        self.norm_input  = nn.LayerNorm(self.embedding_dim)
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
                self.slots_loc = nn.Sequential(
                                nn.Linear(self.embedding_dim, self.key_dim),
                                nn.GELU(),
                                nn.Linear(self.key_dim, self.key_dim),
                            )

                self.slots_logscale = nn.Sequential(
                                nn.Linear(self.embedding_dim, self.key_dim),
                                nn.GELU(),
                                nn.Linear(self.key_dim, self.key_dim),
                            )
        else:
            self.init_slot_transformation = nn.Sequential(
                            nn.Linear(input_dim , hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, self.key_dim),
                        )

      
        nconditions = args.nconditions
        if (nconditions > 0) and (args.model == 'SSAU'):
            self.conditioning = nn.Linear(self.key_dim + nconditions, self.key_dim)


        # ===========================================
        # encoder postional embedding with linear transformation
        res = int(args.enc_arch.split(',')[-1].split('b')[0])
        ntokens = res**2
        self.encoder_norm        = nn.LayerNorm([ntokens, self.embedding_dim])
        self.encoder_position    = SoftPositionEmbed(self.embedding_dim, (res, res))
        self.encoder_feature_mlp = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                                nn.GELU(),
                                                nn.Linear(self.embedding_dim, self.embedding_dim),
                                                nn.GELU())


        # ============================================
        # decoder setting
        # decoder positional embeddings
        resolution = (self.init_spatial_resolution, self.init_spatial_resolution)
        self.decoder_position    = SoftPositionEmbed(self.key_dim, resolution)



    def get_stochastic_initial_slots(self, 
                                        features: Optional[Tensor] = None, 
                                        nsamples: int = 1, 
                                        nslots: Optional[int] = None,
                                        device: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        # features: B x channels x res x res
        nsamples = nsamples if features is None else features.shape[0] 
        nslots = nslots if nslots else self.nslots


        if self.variational_slots:
            if features is None:
                slots_loc = torch.zeros(nsamples, self.key_dim, device = device)
                slots_logscale = torch.ones(nsamples, self.key_dim, device = device)
            else:
                features = features.flatten(-2, -1).mean(-1) 
                slots_loc = self.slots_loc(features)
                slots_logscale = self.slots_logscale(features)
        else:
            slots_loc = self.slots_loc.expand(nsamples, -1)
            slots_logscale = self.slots_logscale.expand(nsamples, -1)


        slots = slots_loc.unsqueeze(1) +\
                (0.5*slots_logscale).unsqueeze(1).exp() * torch.randn(
                    nsamples, nslots, self.key_dim, device=device
                )

        return (slots, slots_loc.unsqueeze(-1).unsqueeze(-1), slots_logscale.unsqueeze(-1).unsqueeze(-1))



    def get_deterministic_initial_slots(self, 
                                            features_loc: Tensor, 
                                            features_logscale: Tensor,
                                            nslots: Optional[int] = None) -> Tensor:
        # features_loc: B x channels x res x res
        nslots = nslots if nslots else self.nslots

        features_loc = features_loc.flatten(-2, -1).permute(0, 2, 1)
        features_logscale = features_logscale.flatten(-2, -1).permute(0, 2, 1)

        b, n, c = features_loc.shape
        xslot = (features_loc.unsqueeze(1) + (0.5*features_logscale).exp().unsqueeze(1) * torch.randn(
                                    b, nslots, n, c, device=x.device)).mean(2)

        return self.init_slot_transformation(xslot)



    def encoder_transformation(self, features: Tensor) -> Tensor:
        #features: B x W x H x C
        features = self.encoder_position(features).permute(0, 2, 3, 1)
        features = torch.flatten(features, 1, 2)
        features = self.encoder_norm(features)
        features = self.encoder_feature_mlp(features)
        return features

    

    def decoder_transformation(self, slots: Tensor) -> Tensor:
        # features: B x nslots x dim
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(2).unsqueeze(3) # (B*nslots) x dim
        features = slots.repeat((1, 1, self.init_spatial_resolution, self.init_spatial_resolution))
        features = self.decoder_position(features)
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

        if self.use_routing:
            slots = self.gru(
                rearrange(slots, 'b n d -> (b n) d'),
                rearrange(slots_prev, 'b n d -> (b n) d')
            )

            slots = slots.reshape(-1, self.nslots, self.key_dim)

        slots = slots + self.slot_transformation(self.norm_pre_ff(slots))

        return slots, attn_vis



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

        if hasattr(self, 'conditioning'):
            _, num_slots, _ = slots.shape
            slots = self.conditioning(torch.cat(slots, properties[:, :num_slots, :], dim = 2))


        # ========================
        tokens = self.encoder_transformation(inputs)
        tokens = self.norm_input(tokens)        


        k, v = self.to_k(tokens), self.to_v(tokens)

            
        if self.use_routing:
            for _ in range(self.niters):
                slots, attn = self.step(slots, k, v)

            if self.implicit: 
                slots, attn = self.step(slots.detach(), k, v)
        else:
            slots, attn = self.step(slots, k, v)


        return self.decoder_transformation(slots), attn, slot_loss


class SlotAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.vr = "light" if "ukbb" in args.hps else None  # hacky
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        self.slot_attention = SlotAttentionWithPositions(args)

        variational_models = ['VSA', 'VASA', 'SSA', 'SSAU']
        bottleneck = args.channels[-1]//args.bottleneck

        self.variational_latents = args.model in variational_models

        if self.variational_latents:    

            self.posterior = Block(
                            args.channels[-1],
                            bottleneck,
                            2 * args.channels[-1],
                            kernel_size=k,
                            residual=False,
                            version=args.vr,
                        )

            self.zprior_type = args.zprior.lower()
            self.learn_prior  = args.learn_prior  

            self.latent_dim = args.channels[-1]
            self.res = res = int(args.enc_arch.split(',')[-1].split('b')[0])

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
                
                if self.lean_prior:
                    self.mu_p_z = nn.Parameter(torch.zeros(1, self.latent_dim, res, res), 
                                                        requires_grad=True)
                    nn.init.xavier_normal_(self.mu_p_z)

                    self.logscale_p_z = nn.Parameter(torch.ones(1, self.latent_dim, res, res), 
                                                        requires_grad=True)
                    nn.init.xavier_normal_(self.logscale_p_z)
                else:
                    self.register_buffer("mu_p_z", torch.zeros(1, self.latent_dim, res, res))
                    self.register_buffer("logscale_p_z", torch.ones(1, self.latent_dim, res, res))
           


        self.projection = nn.Conv2d(
                            args.channels[0], 
                            args.input_channels + 1, 
                            kernel_size=1, stride=1
                        )

        if args.x_like.split("_")[1] == "dgauss":
            self.likelihood = DGaussNet(args)
        else:
            NotImplementedError(f"{args.x_like} not implemented.")
        
        self.free_bits = args.kl_free_bits
        self.register_buffer("log2", torch.tensor(2.0).log())



    def _mixture_kl_loss(self, z: Tensor, loc: Tensor, logscale: Tensor):
        # loc, scale: B x ds
        # slots: B x K x ds
        b, k, c = z.shape

        z = z.view(-1, z.shape[-1]) #treat each slot independent from others
        loc = loc.view(-1, z.shape[-1])
        logscale = logscale.view(-1, z.shape[-1])

        scale = torch.exp(0.5 * logscale)
        q_z = dist.Normal(loc, scale)

        std_pz_c = torch.exp(0.5 *self.logscale_p_z)
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

        
        KL_z = (log_prob_E_qz_x - log_prob_E_pz_c).view(b, c, k, 1)
        KL_c = (log_prob_E_q_c_x - log_prob_E_p_c).view(b, c, k, 1)


        KL = KL_z + KL_c 
   
        return KL 
        

    
    def get_latents(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        loc, logscale = self.posterior(h)
        z = loc + (0.5*logscale).exp()*torch.randn_like(loc)

        if self.zprior == 'gmm':
            kl_loss = self._mixture_kl_loss(z, loc, logscale)
        else:
            kl_loss = gaussian_kl(loc, logscale, self.mu_p_z, self.logscale_p_z)

        return z, loc, logscale, kl_loss



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
            h, h_loc, h_logscale, kl_loss = self.get_latents(h)
            stats.append(dict(kl=kl_loss))


        zs, attn, slot_kl_loss = self.slot_attention(h, h_loc, h_logscale, properties, num_slots)
        stats.append(dict(kl=slot_kl_loss))

        h = self.decoder(zs)
        xh, recons, masks = self.slotpixel_competition(x.shape[0], h)
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
            z_dist = dist.Normal(loc = torch.zeros_like(self.latent_dim, self.res, self.res), 
                                    scale= torch.ones_like(self.latent_dim, self.res, self.res))
            z_samples = z_dist.sample((nsamples,)).to(device)

            return z_samples

        if not self.gmm:
            h = __vae_latents__()
        else:
            if self.num_components == 1:
                h = __vae_latents__()
            
            if not per_component:
                l_p_c = torch.log_softmax(self.pi_p_c, dim=-1)
                p_c = td.one_hot_categorical.OneHotCategorical(logits=l_p_c)
                c_samples = p_c.sample((nsamples * self.res * self.res, )).to(device)

                loc = c_samples @ self.mu_p_z
                scale = c_samples @ torch.exp(0.5 * self.log_sigma_square_p_z)
                pz_c = td.normal.Normal(loc=loc, scale=scale)
                h = pz_c.sample().to(device)
                h = h.view(nsamples, self.latent_dim, self.res, self.res)
            else:
                c_samples = torch.tensor(np.eye(self.num_components, dtype=np.float32)).to(device)
                loc = c_samples @ self.mu_p_z
                scale = c_samples @ torch.exp(0.5 * self.log_sigma_square_p_z)
                pz_c = td.normal.Normal(loc=loc, scale=scale)
                h = pz_c.sample((nsamples * self.res * self.res,))
                h = h.view(nsamples, self.latent_dim, self.res, self.res)

        h_loc, h_logscale = None, None 
        zs, attn_maps, _ = self.slot_attention(h, h_loc, h_logscale, properties, num_slots)

        latents = self.decoder(zs)
        xh, recons, masks = self.slotpixel_competition(bs, latents)
        return self.likelihood.sample(xh, return_loc, t=t), recons, masks, attn_maps



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
            latents, h_loc, h_logscale, kl_loss = self.get_latents(latents)


        zs, attn_maps, _ = self.slot_attention(latents, h_loc, h_logscale, properties, num_slots)

        latents = self.decoder(zs)
        xh, recons, masks = self.slotpixel_competition(bs, latents)
        return self.likelihood.sample(xh, t=t), recons, masks, attn_maps