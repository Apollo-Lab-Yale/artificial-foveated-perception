from collections import OrderedDict
from typing import List

import torch
from torch import nn


class FrozenCLIPTextEncoder(object):
    """
    Frozen CLIP text encoder with a CPU-side LRU cache.
    """

    def __init__(self,
                 model_name='ViT-B/32',
                 clip_device='cpu',
                 cache_size=4096,
                 default_prompt='A generic robotic manipulation task.'):
        super().__init__()
        self.model_name = model_name
        self.clip_device = clip_device
        self.cache_size = max(0, int(cache_size))
        self.default_prompt = default_prompt

        try:
            import clip
        except Exception as exc:
            raise ImportError(
                'CLIP is required for text conditioning. '
                'Install with: pip install git+https://github.com/openai/CLIP.git'
            ) from exc

        self._clip = clip
        model, _ = clip.load(model_name, device=clip_device, jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        with torch.no_grad():
            probe_tokens = clip.tokenize([self.default_prompt]).to(clip_device)
            probe_embed = model.encode_text(probe_tokens).float()
            probe_embed = probe_embed / probe_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.embed_dim = int(probe_embed.shape[-1])

        self._cache = OrderedDict()

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return self.default_prompt
        text = str(text).strip()
        if len(text) == 0:
            return self.default_prompt
        return text

    @torch.no_grad()
    def encode_texts(self, texts: List[str], out_device: torch.device):
        if len(texts) == 0:
            return torch.empty(0, self.embed_dim, device=out_device, dtype=torch.float32)

        norm_texts = [self._normalize_text(t) for t in texts]
        out = [None] * len(norm_texts)

        uncached_texts = []
        uncached_indices = []

        for i, txt in enumerate(norm_texts):
            if txt in self._cache:
                out[i] = self._cache[txt]
                self._cache.move_to_end(txt)
            else:
                uncached_texts.append(txt)
                uncached_indices.append(i)

        if len(uncached_texts) > 0:
            token_batch = self._clip.tokenize(uncached_texts).to(self.clip_device)
            feats = self.model.encode_text(token_batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            feats_cpu = feats.detach().cpu()

            for local_i, global_i in enumerate(uncached_indices):
                txt = uncached_texts[local_i]
                feat_cpu = feats_cpu[local_i]
                out[global_i] = feat_cpu
                if self.cache_size > 0:
                    self._cache[txt] = feat_cpu
                    self._cache.move_to_end(txt)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)

        out_tensor = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in out], dim=0)
        return out_tensor.to(out_device, dtype=torch.float32)


class TextConditioningAdapter(nn.Module):
    """
    Episode-level FiLM-style adapter for decoder states.
    """

    def __init__(self, hidden_dim, text_embed_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)
        self.confidence = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, query_states, text_embed, has_text=None):
        text_h = self.text_proj(text_embed)

        gamma = torch.tanh(self.gamma(text_h)).view(text_h.shape[0], 1, 1, self.hidden_dim)
        beta = self.beta(text_h).view(text_h.shape[0], 1, 1, self.hidden_dim)

        confidence = torch.sigmoid(self.confidence(text_h)).view(text_h.shape[0], 1, 1, 1)
        if has_text is not None:
            has_text = has_text.float().view(text_h.shape[0], 1, 1, 1)
            confidence = confidence * (0.2 + 0.8 * has_text)

        delta = confidence * (query_states * gamma + beta)
        return query_states + self.dropout(delta)
