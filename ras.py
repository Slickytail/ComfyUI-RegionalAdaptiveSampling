from dataclasses import dataclass
from types import MethodType
from time import sleep

import torch
from torch import nn
from torch import Tensor
from einops import rearrange

from comfy.ldm.flux.model import Flux
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, LastLayer
from comfy.ldm.modules.attention import optimized_attention


def apply_pe(x: Tensor, pe: Tensor) -> Tensor:
    """
    The PE application from flux.math.attention, removed, so that we can cache the keys post-PE
    """
    shape = x.shape
    dtype = x.dtype
    x = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x = (pe[..., 0] * x[..., 0] + pe[..., 1] * x[..., 1]).reshape(*shape).to(dtype)

    return x


@dataclass
class RASConfig:
    start_step: int = 4
    hydrate_every: int = 5
    sample_ratio: float = 0.5
    starvation_scale: float = 0.1
    high_ratio: float = 1.0


class RASManager:
    """
    Coordinates the live indices, metrics, and model wrapping.
    """

    def __init__(self, config: RASConfig):
        self.flipped_img_txt = False
        self.timestep: int = 0
        self.n_txt: int = 0
        self.n_img: int = 0
        self.cached_output: Tensor | None = None
        self.live_txt_indices: Tensor | None = None
        self.live_img_indices: Tensor | None = None
        self.drop_count: torch.Tensor | None = None
        self.config = config
        self.patch_size: list[int]
        self.model: Flux | HunyuanVideo
        assert (
            self.config.high_ratio >= 0 and self.config.high_ratio <= 1
        ), "High ratio should be in the range of [0, 1]"

    def wrap_layer(
        self, layer: DoubleStreamBlock | SingleStreamBlock | LastLayer, first=False
    ):
        if isinstance(layer, DoubleStreamBlock):
            wrapped = DoubleStreamBlockWrapper(layer, self, first)
        elif isinstance(layer, SingleStreamBlock):
            wrapped = SingleStreamBlockWrapper(layer, self)
        elif isinstance(layer, LastLayer):
            wrapped = LastLayerWrapper(layer, self)
        else:
            raise TypeError(f"Can't wrap layer of type {layer.__class__.__name__}")
        return wrapped

    def wrap_model(self, model: Flux | HunyuanVideo):
        self.model = model
        if isinstance(model, Flux):
            self.patch_size = [model.patch_size, model.patch_size]
        elif isinstance(model, HunyuanVideo):
            self.patch_size = model.patch_size
        else:
            raise TypeError(f"Can't wrap model of type {model.__class__.__name__}")
        # wrap the single and double blocks to have caching
        for i, v in enumerate(model.double_blocks):
            if i == 0:
                self.flipped_img_txt = v.flipped_img_txt
                model.double_blocks[i] = self.wrap_layer(v, first=True)
            else:
                model.double_blocks[i] = self.wrap_layer(v)
        for i, v in enumerate(model.single_blocks):
            model.single_blocks[i] = self.wrap_layer(v)

        # wrap the forward_orig method, to be able to get the timestep
        forward_orig = model.forward_orig

        def new_forward(_self, *args, **kwargs):
            transformer_options = args[-1]
            self.timestep = self.timestep_from_sigmas(
                transformer_options["sigmas"], transformer_options["sample_sigmas"]
            )
            return forward_orig(*args, **kwargs)

        model.forward_orig = MethodType(new_forward, model)

        # wrap the last_layer, to be able to read the output, and replace the missing (cached) tokens
        model.final_layer = self.wrap_layer(model.final_layer)

    @staticmethod
    def timestep_from_sigmas(sigmas: Tensor, sample_sigmas: Tensor):
        # we assume that one element of sample_sigmas is exactly equal to sigmas
        # but we'll still check explicitly, using an argmin, in case of some loss of precision
        s = sigmas.item()
        i = torch.argmin(torch.abs(sample_sigmas - s).flatten())
        return int(i.item())

    def skip_ratio(self, timestep: int) -> float:
        if timestep < self.config.start_step or (
            timestep % self.config.hydrate_every == 0
        ):
            return 1
        return 1.0 - self.config.sample_ratio

    def select_indices(self, diff: Tensor, timestep: int):
        if isinstance(self.model, Flux):
            # b (h w) (c ph pw) = model_out.shape
            metric = rearrange(
                diff,
                "b s (c ph pw) -> b s ph pw c",
                ph=self.patch_size[0],
                pw=self.patch_size[1],
            )
            metric = torch.std(metric, dim=-1).mean((-1, -2))
        elif isinstance(self.model, HunyuanVideo):
            # b (h w) (c ph pw) = model_out.shape
            metric = rearrange(
                diff,
                "b s (c pt ph pw ) -> b s pt ph pw c",
                pt=self.patch_size[0],
                ph=self.patch_size[1],
                pw=self.patch_size[2],
            )
            metric = torch.std(metric, dim=-1).mean((-1, -2, -3))
        else:
            raise TypeError("Unknown latent type!")
        # hmm, how do we deal with batch size != 1 here?
        # that might be a problem
        metric = metric.flatten()
        if self.drop_count is None:
            self.drop_count = torch.zeros(
                metric.shape, dtype=torch.int, device=diff.device
            )
        # hmm, what if we do a gaussian blur or some sort of spatial lowpass, to improve the spatial continuity of the patches?
        metric *= torch.exp(self.config.starvation_scale * self.drop_count)
        indices = torch.sort(metric, dim=-1, descending=False).indices
        skip_ratio = self.skip_ratio(timestep)
        if skip_ratio >= 0.99:
            # we're not dropping anything -- remove the live_indices
            # we use the value None to indicate a full hydrate
            self.live_img_indices = None
        else:
            low_bar = int(skip_ratio * len(metric) * (1 - self.config.high_ratio))
            high_bar = int(skip_ratio * len(metric) * self.config.high_ratio)
            cache_indices = torch.cat([indices[:low_bar], indices[-high_bar:]])
            self.live_img_indices = indices[low_bar:-high_bar]
            self.drop_count[cache_indices] += 1
        # TODO: for now we keep all txt tokens
        # in the future, we can probably do something like randomly keep a fraction of them
        self.live_txt_indices = torch.arange(
            0, self.n_txt, dtype=torch.int, device=diff.device
        )

    def live_indices(self):
        if self.live_img_indices is None or self.live_txt_indices is None:
            return None
        if self.flipped_img_txt:
            return torch.cat(
                (self.live_img_indices, self.live_txt_indices + self.n_img)
            )
        else:
            return torch.cat(
                (self.live_txt_indices, self.live_img_indices + self.n_txt)
            )


class LastLayerWrapper(nn.Module):
    """
    Same as the LastLayer, but reports its output to a manager.
    """

    def __init__(self, original: LastLayer, manager: RASManager):
        super().__init__()
        self.original = original
        self.manager = manager

    def forward(self, x, vec) -> Tensor:
        # call the original first
        output = self.original(x, vec)
        # if we were fully hydrating, fill the cache
        if self.manager.live_indices() is None or self.manager.cached_output is None:
            self.manager.cached_output = output.clone()
        else:
            self.manager.cached_output[..., self.manager.live_img_indices, :] = output
        self.manager.select_indices(self.manager.cached_output, self.manager.timestep)
        return self.manager.cached_output


class DoubleStreamBlockWrapper(nn.Module):
    """
    Same as the DoubleStreamBlock, but uses a RASManager and RASCache to do KV caching.
    """

    def __init__(self, original: DoubleStreamBlock, manager: RASManager, first=False):
        super().__init__()
        self.block = original
        self.manager = manager
        self.k_cache: torch.Tensor
        self.v_cache: torch.Tensor
        self.first = first

    def forward(self, img, txt, vec, pe, attn_mask=None):
        # RAS: if this is the first doublestreamblock, then we should drop some of the img and txt tokens
        idx = self.manager.live_indices()
        if self.first:
            self.manager.n_txt = txt.shape[1]
            self.manager.n_img = img.shape[1]

            img_idx = self.manager.live_img_indices
            txt_idx = self.manager.live_txt_indices
            if idx is not None:
                img = img[..., img_idx, :]
                txt = txt[..., txt_idx, :]

        img_mod1, img_mod2 = self.block.img_mod(vec)
        txt_mod1, txt_mod2 = self.block.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.block.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.block.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(
            img_qkv.shape[0], img_qkv.shape[1], 3, self.block.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.block.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.block.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.block.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(
            txt_qkv.shape[0], txt_qkv.shape[1], 3, self.block.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.block.txt_attn.norm(txt_q, txt_k, txt_v)

        # RAS: KV Cache and Attention Call
        # select part of the PE
        if idx is not None:
            pe = pe[:, :, idx]

        # create queries, keys, and values
        if self.block.flipped_img_txt:
            queries = apply_pe(torch.cat((img_q, txt_q), dim=2), pe)
            keys = apply_pe(torch.cat((img_k, txt_k), dim=2), pe)
            values = torch.cat((img_v, txt_v), dim=2)
        else:
            queries = apply_pe(torch.cat((txt_q, img_q), dim=2), pe)
            keys = apply_pe(torch.cat((txt_k, img_k), dim=2), pe)
            values = torch.cat((txt_v, img_v), dim=2)

        # fill in the KV cache
        if idx is None:
            self.k_cache = keys
            self.v_cache = values
        else:
            self.k_cache[..., idx, :] = keys
            self.v_cache[..., idx, :] = values
        # actual attention call
        attn = optimized_attention(
            queries,
            self.k_cache,
            self.v_cache,
            img_q.shape[1],
            skip_reshape=True,
            mask=attn_mask,
        )
        if self.block.flipped_img_txt:
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        else:
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        # End of RAS code

        # calculate the img blocks
        img = img + img_mod1.gate * self.block.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.block.img_mlp(
            (1 + img_mod2.scale) * self.block.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt blocks
        txt += txt_mod1.gate * self.block.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.block.txt_mlp(
            (1 + txt_mod2.scale) * self.block.txt_norm2(txt) + txt_mod2.shift
        )

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlockWrapper(nn.Module):
    """
    Same as the SingleStreamBlock, but uses a RASManager and RASCache to do KV caching.
    """

    def __init__(self, original: SingleStreamBlock, manager: RASManager):
        super().__init__()
        self.block = original
        self.manager = manager
        self.k_cache: torch.Tensor
        self.v_cache: torch.Tensor

    def forward(self, x, vec, pe, attn_mask=None):
        idx = self.manager.live_indices()
        mod, _ = self.block.modulation(vec)
        qkv, mlp = torch.split(
            self.block.linear1((1 + mod.scale) * self.block.pre_norm(x) + mod.shift),
            [3 * self.block.hidden_size, self.block.mlp_hidden_dim],
            dim=-1,
        )

        q, k, v = qkv.view(
            qkv.shape[0], qkv.shape[1], 3, self.block.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        q, k = self.block.norm(q, k, v)

        # RAS: KV Cache
        if idx is not None:
            pe = pe[:, :, idx]
        q = apply_pe(q, pe)
        k = apply_pe(k, pe)
        # full hydrate
        if idx is None:
            self.k_cache = k
            self.v_cache = v
        # partial update
        else:
            self.k_cache[..., idx, :] = k
            self.v_cache[..., idx, :] = v
        attn = optimized_attention(
            q,
            self.k_cache,
            self.v_cache,
            q.shape[1],
            skip_reshape=True,
            mask=attn_mask,
        )
        # End of RAS code
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.block.linear2(torch.cat((attn, self.block.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x
