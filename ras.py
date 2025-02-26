from uuid import uuid4 as uuid

import torch
from torch import nn
from torch import Tensor

from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
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


class RASCache(nn.Module):
    """
    KV Cache container.
    TODO: at the end of the project, decide if this class can be removed (and the *Wrappers given simply a k_cache and v_cache)
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.k_cache: Tensor
        self.v_cache: Tensor


class RASManager:
    """
    Coordinates the live indices, metrics, and model wrapping.
    """

    def __init__(self):
        self.caches = {}
        self.live_indices: torch.IntTensor

    def wrap_layer(self, layer: DoubleStreamBlock | SingleStreamBlock):
        layer_id = uuid()
        self.caches[layer_id] = RASCache(layer.hidden_size)
        if isinstance(layer, DoubleStreamBlock):
            wrapped = DoubleStreamBlockWrapper(layer, self, self.caches[layer_id])
        elif isinstance(layer, SingleStreamBlock):
            wrapped = SingleStreamBlockWrapper(layer, self, self.caches[layer_id])
        else:
            raise TypeError(f"Can't wrap layer of type {layer.__class__.__name__}")

        return wrapped


class DoubleStreamBlockWrapper(nn.Module):
    """
    Same as the DoubleStreamBlock, but uses a RASManager and RASCache to do KV caching.
    """

    def __init__(
        self, original: DoubleStreamBlock, manager: RASManager, cache: RASCache
    ):
        self.block = original
        self.manager = manager
        self.cache = cache

    def forward(self, args, _):
        # we get passed the original args
        img = args["img"]
        txt = args["txt"]
        vec = args["vec"]
        pe = args["pe"]
        attn_mask = args["attention_mask"]

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
        if self.block.flipped_img_txt:
            self.cache.k_cache[..., self.manager.live_indices, :] = apply_pe(
                torch.cat((img_k, txt_k), dim=2), pe
            )
            self.cache.v_cache[..., self.manager.live_indices, :] = torch.cat(
                (img_v, txt_v), dim=2
            )
            attn = optimized_attention(
                apply_pe(torch.cat((img_q, txt_q), dim=2), pe),
                self.k_cache,
                self.v_cache,
                img_q.shape[1],
                skip_reshape=True,
                mask=attn_mask,
            )
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        else:
            self.cache.k_cache[..., self.manager.live_indices, :] = apply_pe(
                torch.cat((txt_k, img_k), dim=2), pe
            )
            self.cache.v_cache[..., self.manager.live_indices, :] = torch.cat(
                (txt_v, img_v), dim=2
            )
            attn = optimized_attention(
                apply_pe(torch.cat((txt_q, img_q), dim=2), pe),
                self.k_cache,
                self.v_cache,
                img_q.shape[1],
                skip_reshape=True,
                mask=attn_mask,
            )

            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        # End of RAS code

        # calculate the img blocks
        img = img + img_mod1.gate * self.block.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.block.img_mlp(
            (1 + img_mod2.scale) * self.block.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
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

    def __init__(
        self, original: SingleStreamBlock, manager: RASManager, cache: RASCache
    ):
        self.block = original
        self.manager = manager
        self.cache = cache

    def forward(self, args, _):
        # we get passed the original args
        x = args["img"]
        vec = args["vec"]
        pe = args["pe"]
        attn_mask = args["attention_mask"]

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
        self.cache.k_cache[..., self.manager.live_indices, :] = apply_pe(k, pe)
        self.cache.v_cache[..., self.manager.live_indices, :] = v
        attn = optimized_attention(
            apply_pe(q, pe),
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
