"""
Unified backbone interface for SAE feature explorer inference.

Provides ``load_batched_backbone()`` — a unified loader for the batched
precompute / extraction scripts. (Single-image on-demand inference in the
explorer builds its own forward fn; see demo/explorer/activations.py.)
"""
from __future__ import annotations

import torch
from torchvision import transforms as trn

_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD  = [0.229, 0.224, 0.225]


def _dino_transform(image_size: int) -> trn.Compose:
    return trn.Compose([
        trn.Resize((image_size, image_size),
                   interpolation=trn.InterpolationMode.BICUBIC,
                   antialias=True),
        trn.ToTensor(),
        trn.Normalize(_DINO_MEAN, _DINO_STD),
    ])


def load_batched_backbone(backbone_name: str, layer, device: torch.device):
    """
    Load a vision backbone for batched precompute scripts.

    Returns
    -------
    forward_fn   : callable (batch_tensor) -> (bs, n_tokens, d_hidden)
    d_hidden     : int
    n_reg        : int  (register tokens; 0 for CLIP)
    transform_fn : callable PIL Image -> Tensor (C, H, W)
    """
    use_intermediate = layer is not None

    if backbone_name == 'clip':
        from transformers import CLIPModel, CLIPImageProcessor
        print(f"Loading CLIP ViT-L/14 on {device}...")
        proc  = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float32,
        ).to(device).eval()
        d_hidden = model.config.vision_config.hidden_size  # 1024
        n_reg    = 0

        def transform_fn(img):
            return proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        if use_intermediate:
            _embed   = model.vision_model.embeddings
            _prenorm = model.vision_model.pre_layrnorm
            _layers  = model.vision_model.encoder.layers

            def forward_fn(imgs):
                h = _embed(imgs)
                h = _prenorm(h)
                for i, enc_layer in enumerate(_layers):
                    out = enc_layer(
                        hidden_states=h, attention_mask=None, causal_attention_mask=None,
                    )
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    if i == layer - 1:
                        return h
                return h
        else:
            def forward_fn(imgs):
                return model.vision_model(pixel_values=imgs).last_hidden_state

    elif backbone_name.startswith('dinov2'):
        print(f"Loading DINOv2 ViT-B/14 on {device}...")
        model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14'
        ).to(device).eval()
        d_hidden = 768
        n_reg    = 0

        _dino_xfm = _dino_transform(224)

        def transform_fn(img):
            return _dino_xfm(img)

        if use_intermediate:
            _act = [None]
            def _hook(module, input, output):
                _act[0] = output
            model.blocks[layer].register_forward_hook(_hook)

            def forward_fn(imgs):
                _act[0] = None
                model.forward_features(imgs)
                return _act[0]  # (bs, 1 + n_patches, 768)
        else:
            def forward_fn(imgs):
                out = model.forward_features(imgs)
                cls     = out['x_norm_clstoken'].unsqueeze(1)
                patches = out['x_norm_patchtokens']
                return torch.cat([cls, patches], dim=1)

    else:  # dinov3
        from transformers import AutoModel
        print(f"Loading DINOv3 ViT-L/16 on {device}...")
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.float32,
        ).to(device).eval()
        d_hidden = model.config.hidden_size
        n_reg    = model.config.num_register_tokens

        _dino_xfm = _dino_transform(256)

        def transform_fn(img):
            return _dino_xfm(img)

        def forward_fn(imgs):
            out = model(pixel_values=imgs, output_hidden_states=use_intermediate)
            return out.hidden_states[layer] if use_intermediate else out.last_hidden_state

    layer_desc = f"layer {layer}" if use_intermediate else "final layer"
    print(f"  d_hidden={d_hidden}, register_tokens={n_reg}, extracting from {layer_desc}")
    return forward_fn, d_hidden, n_reg, transform_fn
