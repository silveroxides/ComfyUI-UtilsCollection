"""Native Whisper architecture adapted from OpenAI Whisper (MIT).

Checkpoint names are unchanged. See whisper_assets/LICENSE for attribution.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.model_management import throw_exception_if_processing_interrupted



@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head, operations):
        super().__init__()
        self.n_head = n_head
        self.query = operations.Linear(n_state, n_state)
        self.key = operations.Linear(n_state, n_state, bias=False)
        self.value = operations.Linear(n_state, n_state)
        self.out = operations.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k, v = kv_cache[self.key], kv_cache[self.value]
        if k.shape[0] != q.shape[0]:
            k = k.expand(q.shape[0], -1, -1)
            v = v.expand(q.shape[0], -1, -1)
        # Cached queries occupy the final positions of the complete key sequence.
        if mask is not None:
            offset = k.shape[1] - q.shape[1]
            mask = mask[offset:offset + q.shape[1], :k.shape[1]]
            mask = comfy.ops.cast_to_input(mask, q)
        attention = optimized_attention_for_device(q.device, mask=mask is not None)
        return self.out(attention(q, k, v, self.n_head, mask=mask))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, operations, cross_attention=False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, operations)
        self.attn_ln = operations.LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head, operations) if cross_attention else None
        self.cross_attn_ln = operations.LayerNorm(n_state) if cross_attention else None
        self.mlp = nn.Sequential(operations.Linear(n_state, n_state * 4), nn.GELU(), operations.Linear(n_state * 4, n_state))
        self.mlp_ln = operations.LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        throw_exception_if_processing_interrupted()
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        return x + self.mlp(self.mlp_ln(x))


class AudioEncoder(nn.Module):
    def __init__(self, dims, operations):
        super().__init__()
        self.conv1 = operations.Conv1d(dims.n_mels, dims.n_audio_state, kernel_size=3, padding=1)
        self.conv2 = operations.Conv1d(dims.n_audio_state, dims.n_audio_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", torch.empty(dims.n_audio_ctx, dims.n_audio_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(dims.n_audio_state, dims.n_audio_head, operations) for _ in range(dims.n_audio_layer)])
        self.ln_post = operations.LayerNorm(dims.n_audio_state)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x)).permute(0, 2, 1)
        if x.shape[1:] != self.positional_embedding.shape:
            raise ValueError("Whisper encoder requires a padded 30-second mel window.")
        x = x + comfy.ops.cast_to_input(self.positional_embedding, x)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)


class TextDecoder(nn.Module):
    def __init__(self, dims, operations):
        super().__init__()
        self.token_embedding = operations.Embedding(dims.n_vocab, dims.n_text_state)
        self.positional_embedding = nn.Parameter(torch.empty(dims.n_text_ctx, dims.n_text_state))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(dims.n_text_state, dims.n_text_head, operations, cross_attention=True) for _ in range(dims.n_text_layer)])
        self.ln = operations.LayerNorm(dims.n_text_state)
        self.register_buffer("mask", torch.full((dims.n_text_ctx, dims.n_text_ctx), -torch.inf).triu_(1), persistent=False)

    def forward(self, tokens, xa, kv_cache=None):
        offset = kv_cache[self.blocks[0].attn.key].shape[1] if kv_cache else 0
        x = self.token_embedding(tokens, out_dtype=xa.dtype)
        x = x + comfy.ops.cast_to_input(self.positional_embedding[offset:offset + tokens.shape[-1]], x)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        # Use the embedding's managed weights for the tied output projection too.
        with comfy.ops.CastBiasWeightContext(self.token_embedding, input=x, offloadable=True) as (weight, _):
            return F.linear(x, weight).float()


class Whisper(nn.Module):
    def __init__(self, dims, operations=comfy.ops.manual_cast):
        super().__init__()
        self.dims = dims
        self.compute_dtype = torch.float32
        self.device = torch.device("cpu")
        self.encoder = AudioEncoder(dims, operations)
        self.decoder = TextDecoder(dims, operations)

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)

    def forward(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))

    def install_kv_cache_hooks(self):
        cache, hooks = {}, []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1)
            return cache[module]

        for layer in self.decoder.modules():
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))
        return cache, hooks
