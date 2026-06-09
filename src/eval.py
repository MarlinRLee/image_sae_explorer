import os
import json
import torch

from overcomplete.sae import TopKSAE

from metric import evaluate_sae


def run_evaluation(config, args, device, loader, d_brain, run_suffix=""):
    """
    Load the trained model and run evaluation metrics.

    Parameters
    ----------
    config : dict
        Training/eval configuration (d_model, k_fraction, etc.).
    args : argparse.Namespace
        Must have: output_dir.
    device : str or torch.device
    loader : DeviceDataLoader
    d_brain : int
        Input activation dimension.
    run_suffix : str
        Suffix used when saving the model (e.g. "_d32000_k160").

    Returns
    -------
    dict or None
        Final results, or None if no model was found.
    """
    print(f"\n{'='*60}")
    print("--- Starting Evaluation ---")
    print(f"{'='*60}")

    k = int(config['k_fraction'] * config['d_model'])
    save_path = os.path.join(args.output_dir, f"sae{run_suffix}_state_dict.pth")

    if not os.path.exists(save_path):
        print(f"[Warning] Trained model not found: {save_path}.")
        return None

    sae = TopKSAE(input_shape=d_brain, nb_concepts=config['d_model'], top_k=k, device=device)
    sae.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    sae.to(device)
    sae.eval()

    print("Running evaluation...")
    with torch.inference_mode():
        metrics = evaluate_sae(sae, loader, device)

    final_output = {
        "config": config,
        "metrics": metrics,
    }

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(final_output, indent=2))

    results_path = os.path.join(args.output_dir, f"final_results{run_suffix}_eval_only.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return final_output
