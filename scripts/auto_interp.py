"""Budget-limited Gemini auto-interp for the PRIMARY registry model, synced
with the canonical label JSONs on Hugging Face.

What it does:

  1. **pulls** the current ``_feature_names.json`` + ``_auto_interp.json`` for
     the primary model from the HF dataset repo (the canonical labels edited
     live in the Space);
  2. labels only features that have **neither** a manual name **nor** an
     existing auto-interp label, most-frequently-firing first;
  3. stops once it has spent ~``--budget`` US dollars of Gemini usage, measured
     from the API's reported token counts (not an estimate);
  4. **pushes** the updated ``_auto_interp.json`` (+ ``_authors`` + ``_history``)
     back to the HF dataset repo.

No GPU required. Runs fully local with ``--no-push`` (and without HF_TOKEN it
just skips the pull/push and labels everything unlabeled on disk).

Environment
-----------
    GOOGLE_API_KEY   required — Gemini API key
    HF_TOKEN         required to push (write token for the dataset repo)
    HF_DATASET_REPO  optional — overrides the registry's defaults.hf_data_repo

Pricing
-------
Defaults are Gemini 2.5 Flash Standard tier as of mid-2026
(input $0.10 / 1M tokens, output $0.40 / 1M tokens). Override with
``--input-price`` / ``--output-price`` if the rate card changes; the actual
token counts come from the API, so only the $/token conversion is assumed.
"""

import argparse
import io
import json
import os
import sys
import time

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'demo'))

# `google.genai` is imported lazily (in the call site) so --dry-run and arg
# parsing work in environments without the google-genai package.
from explorer.registry import load_registry


# ---------------------------------------------------------------------------
# Prompt + image helpers. Kept in sync with the explorer's live button
# (demo/explorer/gemini.py) so batch labels match interactive ones; duplicated
# rather than imported because that module pulls in Bokeh, which a headless
# cluster node shouldn't need.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are labeling features of a Sparse Autoencoder (SAE) trained on a "
    "vision transformer. Each SAE feature is a sparse direction in activation "
    "space that fires strongly on certain visual patterns."
)

USER_PROMPT = (
    "The images below are the top maximally-activating images for one SAE feature. "
    "In 2–5 words, give a precise label for the visual concept this feature detects. "
    "Be specific — prefer 'dog snout close-up' over 'dog', or 'brick wall texture' "
    "over 'texture'. "
    "Reply with ONLY the label, no explanation, no punctuation at the end."
)


def _encode_image(path, size=224):
    """Resize an image and return raw JPEG bytes."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _resolve_path(stored_path, image_dir, extra_image_dir):
    """Resolve an image path the same way the explorer app does."""
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    basename = os.path.basename(stored_path)
    for base in filter(None, [image_dir, extra_image_dir]):
        candidate = os.path.join(base, basename)
        if os.path.exists(candidate):
            return candidate
    # Last resort: stored path as-is
    if os.path.exists(stored_path):
        return stored_path
    return None


# ---------------------------------------------------------------------------
# Tiny HF JSON helpers (inlined to avoid importing explorer.persistence, which
# pulls in Bokeh — not needed on a headless cluster node).
# ---------------------------------------------------------------------------

def _fetch_remote_dict(filename, repo, token):
    """Download a JSON sidecar from the HF dataset; {} if absent/unreadable."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        local = hf_hub_download(repo_id=repo, filename=filename,
                                repo_type="dataset", token=token)
        with open(local) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}
    except Exception as e:  # noqa: BLE001
        print(f"  Warning: HF fetch failed for {filename}: {e}")
        return {}


def _save_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _merge_remote_wins(local, remote, is_history):
    """Union local + remote. On key conflict the REMOTE value wins, so a label
    added in the live Space during this run is never clobbered (our new keys
    aren't on the remote, so they're preserved). History lists are concatenated
    and deduped by (label, author, ts)."""
    if not remote:
        return local
    if is_history:
        out = {}
        for k in set(local) | set(remote):
            seen, merged = set(), []
            for e in (local.get(k) or []) + (remote.get(k) or []):
                if not isinstance(e, dict):
                    continue
                sig = (e.get("label"), e.get("author"), e.get("ts"))
                if sig in seen:
                    continue
                seen.add(sig)
                merged.append(e)
            out[k] = merged
        return out
    out = dict(local)
    out.update(remote)
    return out


def _merge_save_upload(path, repo, token, msg):
    """Re-fetch the remote copy, union-merge (remote wins), save locally, upload."""
    from huggingface_hub import upload_file
    fname = os.path.basename(path)
    with open(path) as f:
        local = json.load(f)
    remote = _fetch_remote_dict(fname, repo, token)
    merged = _merge_remote_wins(local, remote, is_history=fname.endswith("_history.json"))
    _save_json_atomic(path, merged)
    upload_file(path_or_fileobj=path, path_in_repo=fname,
                repo_id=repo, repo_type="dataset", token=token,
                commit_message=msg)
    print(f"  Pushed {fname} -> {repo}")


# ---------------------------------------------------------------------------
# Gemini call with token accounting
# ---------------------------------------------------------------------------

def _label_with_usage(client, model, mei_paths, n_images,
                      image_dir, extra_dir, img_size):
    """Return (label_or_None, prompt_tokens, output_tokens)."""
    from google.genai import types
    parts = []
    for p in mei_paths[:n_images]:
        resolved = _resolve_path(p, image_dir, extra_dir)
        if resolved is None:
            continue
        try:
            parts.append(types.Part.from_bytes(
                data=_encode_image(resolved, size=img_size),
                mime_type="image/jpeg"))
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        return None, 0, 0
    parts.append(types.Part.from_text(text=USER_PROMPT))
    resp = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT))
    um = getattr(resp, "usage_metadata", None)
    pin = int(getattr(um, "prompt_token_count", 0) or 0)
    pout = int(getattr(um, "candidates_token_count", 0) or 0)
    label = (resp.text or "").strip().strip(".,;:\"'")
    return (label or None), pin, pout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(_HERE, "..", "configs", "models.yaml"),
                    help="Registry YAML; the primary entry is labeled.")
    ap.add_argument("--data", required=True,
                    help="Local explorer_data .pt for the primary model "
                         "(its basename should match the registry data_file).")
    ap.add_argument("--image-dir", default=None, help="Primary image directory.")
    ap.add_argument("--extra-image-dir", default=None, help="Extra image directory.")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="Stop after spending ~this many USD (default: 10).")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    ap.add_argument("--n-images", type=int, default=6, help="MEIs per feature.")
    ap.add_argument("--img-size", type=int, default=224, help="Square resize before sending.")
    ap.add_argument("--input-price", type=float, default=0.10,
                    help="USD per 1M input tokens (default: 0.10).")
    ap.add_argument("--output-price", type=float, default=0.40,
                    help="USD per 1M output tokens (default: 0.40).")
    ap.add_argument("--hf-repo", default=None,
                    help="HF dataset repo (default: HF_DATASET_REPO env or "
                         "registry defaults.hf_data_repo).")
    ap.add_argument("--sleep", type=float, default=0.1, help="Seconds between calls.")
    ap.add_argument("--save-interval", type=int, default=50,
                    help="Write local JSON every N labels (default: 50).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Pull + count candidates, make no API calls, no push.")
    ap.add_argument("--no-push", action="store_true",
                    help="Label + save locally but don't upload to HF.")
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: GOOGLE_API_KEY is not set.")
    hf_token = os.environ.get("HF_TOKEN")

    reg = load_registry(args.registry)
    primary = reg.primary
    hf_repo = args.hf_repo or os.environ.get("HF_DATASET_REPO") or reg.hf_data_repo
    print(f"Primary model: {primary.id} ({primary.label})")
    print(f"HF dataset repo: {hf_repo or '(none)'}")

    # Canonical sidecar names follow the loader convention: <data stem>_*.json,
    # derived from the registry data_file so they match what's on HF.
    base = os.path.splitext(primary.data_file)[0]
    auto_name    = base + "_auto_interp.json"
    names_name   = base + "_feature_names.json"
    authors_name = base + "_auto_interp_authors.json"
    history_name = base + "_auto_interp_history.json"

    # ---- Pull current labels from HF ----
    print("Pulling current labels from HF ...")
    auto    = _fetch_remote_dict(auto_name,    hf_repo, hf_token) if hf_repo else {}
    names   = _fetch_remote_dict(names_name,   hf_repo, hf_token) if hf_repo else {}
    authors = _fetch_remote_dict(authors_name, hf_repo, hf_token) if hf_repo else {}
    history = _fetch_remote_dict(history_name, hf_repo, hf_token) if hf_repo else {}
    print(f"  remote: {len(names)} manual, {len(auto)} auto-interp labels")

    # ---- Load tensors ----
    print(f"Loading {args.data} ...")
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    image_paths  = d["image_paths"]
    d_model      = int(d["d_model"])
    top_img_idx  = d["top_img_idx"]
    freq         = d["feature_frequency"]
    n_top_stored = top_img_idx.shape[1]

    # ---- Candidates: alive AND missing any label, most-frequent first ----
    labeled = set(auto) | set(names)            # string keys, as on HF
    order = torch.argsort(freq, descending=True).tolist()
    candidates = [f for f in order
                  if freq[f].item() > 0 and str(f) not in labeled]
    print(f"  {len(candidates)} alive, unlabeled feature(s) to consider "
          f"(of {d_model} total)")

    if args.dry_run:
        print(f"[dry-run] would label up to {len(candidates)} features, "
              f"budget ${args.budget:.2f}. No API calls made.")
        return

    from google import genai
    client = genai.Client(api_key=api_key)
    in_rate  = args.input_price / 1e6
    out_rate = args.output_price / 1e6

    spent = 0.0
    n_labeled = n_failed = 0
    work_dir = os.path.dirname(os.path.abspath(args.data))
    auto_path = os.path.join(work_dir, auto_name)

    def _save_local():
        _save_json_atomic(auto_path, {str(k): v for k, v in auto.items()})

    print(f"Labeling (budget ${args.budget:.2f}, "
          f"${args.input_price}/1M in, ${args.output_price}/1M out) ...")
    for feat in candidates:
        if spent >= args.budget:
            print(f"Budget reached (${spent:.4f} >= ${args.budget:.2f}).")
            break
        mei_paths = [image_paths[top_img_idx[feat, j].item()]
                     for j in range(n_top_stored)
                     if top_img_idx[feat, j].item() >= 0]
        if not mei_paths:
            continue
        try:
            label, pin, pout = _label_with_usage(
                client, args.model, mei_paths, args.n_images,
                args.image_dir, args.extra_image_dir, args.img_size)
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            err = str(e)
            print(f"  feat {feat:6d}: ERROR — {err[:140]}")
            if ("PerDay" in err or "per_day" in err.lower()
                    or "RESOURCE_EXHAUSTED" in err):
                print("Quota exhausted — saving progress and stopping.")
                break
            if "NOT_FOUND" in err or "no longer available" in err:
                print("Model unavailable — check --model and stop.")
                break
            continue

        if not label:
            n_failed += 1
            print(f"  feat {feat:6d}: (no images loaded)")
            continue

        cost = pin * in_rate + pout * out_rate
        spent += cost
        auto[str(feat)] = label
        authors[str(feat)] = args.model
        history.setdefault(str(feat), []).append(
            {"label": label, "author": args.model, "ts": _now_iso()})
        n_labeled += 1
        print(f"  feat {feat:6d} (freq {int(freq[feat].item()):>5}): {label}"
              f"   [${spent:.4f} / ${args.budget:.2f}]")

        if n_labeled % args.save_interval == 0:
            _save_local()
            print(f"  [checkpoint] {n_labeled} labels saved locally")
        if args.sleep:
            time.sleep(args.sleep)

    # ---- Final local save ----
    _save_local()
    authors_path = os.path.join(work_dir, authors_name)
    history_path = os.path.join(work_dir, history_name)
    _save_json_atomic(authors_path, {str(k): v for k, v in authors.items()})
    _save_json_atomic(history_path, {str(k): v for k, v in history.items()})

    print(f"\nLabeled {n_labeled} feature(s), {n_failed} failed. "
          f"Spent ~${spent:.4f}.")

    # ---- Push to HF ----
    if args.no_push:
        print("--no-push set; skipping upload.")
        return
    if not (hf_repo and hf_token):
        print("HF_TOKEN / repo not set; skipping upload (local files written).")
        return
    if n_labeled == 0:
        print("No new labels; nothing to push.")
        return
    print(f"Pushing to HF dataset {hf_repo} ...")
    _merge_save_upload(auto_path,    hf_repo, hf_token, f"Auto-interp: +{n_labeled} labels ({primary.id})")
    _merge_save_upload(authors_path, hf_repo, hf_token, f"Auto-interp authors ({primary.id})")
    _merge_save_upload(history_path, hf_repo, hf_token, f"Auto-interp history ({primary.id})")
    print("Done.")


if __name__ == "__main__":
    main()
