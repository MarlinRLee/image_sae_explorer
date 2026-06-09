"""
CLIP text-alignment utilities for SAE feature interpretation.

Key functions:
- load_clip: load a CLIP model + processor.
- compute_text_embeddings: encode text strings into L2-normalised CLIP embeddings.

The precomputed scores can be stored in explorer_data.pt under:
    'clip_text_scores'   : Tensor (n_features, n_vocab)  float16
    'clip_text_vocab'    : list[str]
    'clip_feature_embeds': Tensor (n_features, clip_proj_dim)  float32
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_clip(device: str | torch.device = "cpu", model_name: str = "openai/clip-vit-large-patch14"):
    """
    Load a CLIP model and processor.

    Parameters
    ----------
    device : str or torch.device
    model_name : str
        HuggingFace model ID.  Default matches the ViT-L/14 variant used by
        many vision papers and is a reasonable match for DINOv3-ViT-L/16.

    Returns
    -------
    model : CLIPModel (eval mode, on device)
    processor : CLIPProcessor
    """
    print(f"Loading CLIP ({model_name})...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()
    print(f"  CLIP loaded (d_text={model.config.projection_dim})")
    return model, processor


# ---------------------------------------------------------------------------
# Core alignment computation
# ---------------------------------------------------------------------------

def compute_text_embeddings(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str | torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Encode a list of text strings into L2-normalised CLIP text embeddings.

    Returns
    -------
    Tensor of shape (len(texts), clip_proj_dim), float32, on CPU.
    """
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            # Go through text_model + text_projection directly to avoid
            # version differences in get_text_features() return type.
            text_out = model.text_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
            )
            embeds = model.text_projection(text_out.pooler_output)
            embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu().float())
    return torch.cat(all_embeds, dim=0)  # (n_texts, clip_proj_dim)
