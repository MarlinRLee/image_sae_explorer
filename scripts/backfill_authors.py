"""One-time backfill: create *_authors.json sibling files for every
feature_names.json / auto_interp.json already in the HF dataset repo.

Existing manual labels are attributed to ``--manual-owner`` (default
``Ramnie``); existing auto-interp labels are attributed to
``--auto-interp-owner`` (default ``gemini-2.5-flash``). Authors files
that already exist are left alone unless ``--force`` is passed.

Usage:
    python scripts/backfill_authors.py
    python scripts/backfill_authors.py --manual-owner alice
    python scripts/backfill_authors.py --dry-run
    python scripts/backfill_authors.py --force            # overwrite existing
"""

import argparse
import json
import os
import sys
import tempfile

from huggingface_hub import HfApi, hf_hub_download, upload_file


def _read_token(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"ERROR: HF token file not found: {path}")
    with open(path) as f:
        return f.read().strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default="Ramnie/sae-explorer-data",
                    help="HF dataset repo to backfill")
    ap.add_argument("--manual-owner", default="Ramnie",
                    help="Author string for entries in *_feature_names.json")
    ap.add_argument("--auto-interp-owner", default="gemini-2.5-flash",
                    help="Author string for entries in *_auto_interp.json")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.hf_token"),
                    help="Path to HF write token file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print what would be uploaded; don't push")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite *_authors.json files that already exist")
    args = ap.parse_args()

    token = _read_token(args.token_file)
    api = HfApi(token=token)
    print(f"Listing repo {args.repo} ...")
    repo_files = api.list_repo_files(args.repo, repo_type="dataset")

    label_files = [
        f for f in repo_files
        if f.endswith("_feature_names.json") or f.endswith("_auto_interp.json")
    ]
    print(f"  found {len(label_files)} label JSON file(s)")

    work = []  # (label_filename, authors_filename, owner_string)
    for label_fname in label_files:
        owner = (args.manual_owner if label_fname.endswith("_feature_names.json")
                 else args.auto_interp_owner)
        authors_fname = label_fname.replace(".json", "_authors.json")
        if authors_fname in repo_files and not args.force:
            print(f"  skip {label_fname}: {authors_fname} already exists")
            continue
        work.append((label_fname, authors_fname, owner))

    if not work:
        print("Nothing to do.")
        return

    print(f"\nWill backfill {len(work)} authors file(s):")
    for label_fname, authors_fname, owner in work:
        print(f"  {label_fname}  ->  {authors_fname}  (owner={owner})")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        for label_fname, authors_fname, owner in work:
            local_label = hf_hub_download(args.repo, label_fname,
                                          repo_type="dataset",
                                          local_dir=tmp,
                                          token=token)
            with open(local_label) as f:
                labels = json.load(f)
            authors = {feat_str: owner for feat_str in labels.keys()}
            local_authors = os.path.join(tmp, authors_fname)
            with open(local_authors, "w") as f:
                json.dump(authors, f, indent=2, sort_keys=True)
            print(f"  prepared {authors_fname} ({len(authors)} entries, owner={owner})")
            if args.dry_run:
                continue
            upload_file(
                path_or_fileobj=local_authors,
                path_in_repo=authors_fname,
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
                commit_message=f"Backfill authors for {label_fname} (owner={owner})",
            )
            print(f"    pushed to {args.repo}/{authors_fname}")

    if args.dry_run:
        print("\nDry-run only; nothing was uploaded.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
