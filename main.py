"""full pipeline for the latent factor model comparison study. usage: python main.py [--device cpu|cuda] [--cache data/raw/data_cache.pkl]"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch


class _Tee:
    """mirrors stdout to a file while still printing to the terminal."""

    def __init__(self, filepath: str) -> None:
        self._file   = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()


def set_seeds(seed: int) -> None:
    """fixes random seeds across all libraries for reproducibility. params: {seed: int}. returns None."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN has a slight performance cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent factor model comparison: PCA vs AE vs CAE"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Torch device for neural network training (default: cpu)",
    )
    parser.add_argument(
        "--cache", default="data/raw/data_cache.pkl",
        help="Path to the data cache file (created on first run)",
    )
    parser.add_argument(
        "--figures-dir", default="paper/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Load pre-trained checkpoints instead of retraining",
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Train S models per config with different seeds for ensembling "
             "and stability reporting (default: False, single-seed run)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Override default seeds list (e.g. --seeds 42 123 456)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # step 0: timestamped run directory
    run_dir    = os.path.join("results", datetime.now().strftime("%m-%d-%Y-%H%M"))
    plots_dir  = os.path.join(run_dir, "plots")
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    tee = _Tee(os.path.join(run_dir, "terminal_output.txt"))

    # step 1: random seeds
    set_seeds(args.seed)
    print(f"Run directory: {run_dir}")
    print(f"Random seed:   {args.seed}")
    print(f"Device:        {args.device}")

    from src.train import train_all_models, K_GRID, LAMBDA_LIN_GRID, LAMBDA_NONLIN_GRID, SEEDS
    print(f"K grid:          {K_GRID}")
    print(f"λ_lin grid:      {LAMBDA_LIN_GRID}")
    print(f"λ_nonlin grid:   {LAMBDA_NONLIN_GRID}")
    print(f"Total CAE configs: {len(K_GRID) * len(LAMBDA_LIN_GRID) * len(LAMBDA_NONLIN_GRID)}")

    seeds = args.seeds if args.seeds is not None else SEEDS

    # step 2: data
    from src.data import load_data
    data = load_data(cache_path=args.cache)
    splits = data["splits"]

    for split_name, split_data in splits.items():
        ret   = split_data["returns"]
        chars = split_data["chars"]
        print(f"  {split_name:5s}: returns {ret.shape}  chars {chars.shape}")

    # step 3: training
    if args.skip_train:
        print("\nSkipping training — loading checkpoints (not implemented yet).")
        tee.close()
        sys.exit(0)

    from src.evaluate import run_evaluation, format_paper_story

    if args.multi_seed:
        print(f"\nMulti-seed mode: {len(seeds)} seeds = {seeds}")
        models = train_all_models(splits, k_list=K_GRID, device=args.device,
                                  multi_seed=True, seeds=seeds, run_dir=run_dir)

        from src.ensemble import build_ensembles, compute_seed_stability, validate_ensemble_improvement

        print("\nBuilding ensemble models …")
        ensemble_models = build_ensembles(models)

        print("\nValidating ensemble improvement …")
        validate_ensemble_improvement(splits, models, ensemble_models,
                                      results_dir=run_dir)

        print("\nComputing seed stability metrics …")
        stability_df = compute_seed_stability(splits, models, results_dir=run_dir)

        # step 4: evaluation (on ensembled predictions)
        summary, sig_results = run_evaluation(
            splits, ensemble_models,
            figures_dir=plots_dir,
            results_dir=run_dir,
            stability_df=stability_df,
        )

    else:
        print(f"\nSingle-seed mode: seed={args.seed}")
        set_seeds(args.seed)
        models = train_all_models(splits, k_list=K_GRID, device=args.device,
                                  run_dir=run_dir)

        # step 4: evaluation
        summary, sig_results = run_evaluation(splits, models,
                                              figures_dir=plots_dir,
                                              results_dir=run_dir)

    # step 5: print results in paper narrative order
    print("\n" + format_paper_story(summary, sig_results))

    tee.close()


if __name__ == "__main__":
    main()
