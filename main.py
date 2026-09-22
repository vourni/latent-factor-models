"""Run the latent factor comparison. See README.md and python main.py --help."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch


class _Tee:
    """Mirror output without taking ownership of the original stream."""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.log.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PCA, IPCA, AE, and residual conditional autoencoders")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--cache", default="data/raw/data_cache_v2.pkl")
    parser.add_argument("--allow-legacy-cache", action="store_true",
                        help="Explicitly allow historical characteristics for diagnostics")
    parser.add_argument("--run-dir", help="New output directory; existing directory required with --skip-train")
    parser.add_argument("--figures-dir", help="Override the default RUN_DIR/plots directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true", help="Ensemble all three CAE variants")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seeds for --multi-seed")
    parser.add_argument("--k", type=int, nargs="+", help="Factor dimensions, default: 2 3 5 8 10")
    parser.add_argument("--lambda-lin", type=float, nargs="+")
    parser.add_argument("--lambda-nonlin", type=float, nargs="+")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads (default: 1)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Re-evaluate a trusted checkpoint from --run-dir with the same data")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Small offline synthetic run; not empirical research results")
    args = parser.parse_args(argv)
    if args.skip_train and not args.run_dir:
        parser.error("--skip-train requires --run-dir containing models/checkpoint.pt")
    if args.seeds and not args.multi_seed and not args.skip_train:
        parser.error("--seeds requires --multi-seed")
    for name in ("threads", "max_epochs", "patience", "bootstrap_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("k", "lambda_lin", "lambda_nonlin", "seeds"):
        values = getattr(args, name)
        if values is not None:
            if len(values) != len(set(values)) or not all(np.isfinite(v) for v in values):
                parser.error(f"--{name.replace('_', '-')} must contain distinct finite values")
            if name == "k" and any(v <= 0 for v in values):
                parser.error("--k must contain positive integers")
            if name.startswith("lambda") and any(v < 0 for v in values):
                parser.error("Regularization weights cannot be negative")
    return args


def synthetic_data(seed=42):
    """Small known factor process for testing the entire offline pipeline."""
    import pandas as pd
    from src.data import time_split
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2008-01-31", "2024-12-31", freq="ME")
    n_assets, n_chars, k = 40, 6, 2
    chars = rng.uniform(-1, 1, (n_assets, len(dates), n_chars)).astype(np.float32)
    gamma = rng.normal(0, 0.4, (n_chars, k))
    beta = chars @ gamma
    beta[:, :, 0] += 0.2 * chars[:, :, 0] ** 2
    factors = rng.normal(0.01, 0.035, (len(dates), k))
    returns = np.einsum("ntk,tk->tn", beta, factors) + rng.normal(0, 0.03, (len(dates), n_assets))
    returns = pd.DataFrame(returns, index=dates, columns=[f"S{i:03d}" for i in range(n_assets)])
    return {"returns": returns, "chars": chars, "dates": dates, "tickers": list(returns.columns),
            "splits": time_split(returns, chars), "synthetic": True, "cache_version": "synthetic-1"}


def data_fingerprint(data):
    """Hash data values, labels, and split boundaries used by the checkpoint."""
    digest = hashlib.sha256()
    for name, split in data["splits"].items():
        ret = split["returns"]
        digest.update(name.encode())
        digest.update(str(list(ret.columns)).encode())
        digest.update(ret.index.asi8.tobytes())
        for array in (ret.values, split["chars"]):
            array = np.asarray(array, dtype=np.float64)
            digest.update(str(array.shape).encode())
            digest.update(np.isnan(array).tobytes())
            digest.update(np.nan_to_num(array).tobytes())
    return digest.hexdigest()


def _move_models(models, device):
    for family in ("ae", "cae", "cae_fixed", "cae_nl"):
        for value in models.get(family, {}).values():
            for model in value if isinstance(value, list) else [value]:
                model.device = torch.device(device)
                model.net.to(model.device)


def _save_histories(models, path):
    histories = {}
    for family in ("ae", "cae", "cae_fixed", "cae_nl"):
        for key, value in models.get(family, {}).items():
            for i, model in enumerate(value if isinstance(value, list) else [value]):
                histories[f"{family} {key} seed={getattr(model, 'seed', i)}"] = {
                    "train_losses": model.train_losses, "val_losses": model.val_losses}
    path.write_text(json.dumps(histories, indent=2) + "\n")


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu or an available MPS device.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; use --device cpu.")
    torch.set_num_threads(args.threads)
    set_seeds(args.seed)
    from src import train
    from src.data import load_data
    from src.evaluate import run_evaluation, format_paper_story

    train.K_GRID = args.k or ([2] if args.smoke_test else train.K_GRID)
    train.LAMBDA_LIN_GRID = args.lambda_lin or ([0.001] if args.smoke_test else train.LAMBDA_LIN_GRID)
    train.LAMBDA_NONLIN_GRID = args.lambda_nonlin or ([0.00001] if args.smoke_test else train.LAMBDA_NONLIN_GRID)
    train.MAX_EPOCHS = args.max_epochs or (3 if args.smoke_test else train.MAX_EPOCHS)
    train.PATIENCE = args.patience or (2 if args.smoke_test else train.PATIENCE)
    n_bootstrap = args.bootstrap_samples or (20 if args.smoke_test else 1000)
    seeds = args.seeds or train.SEEDS
    run_dir = Path(args.run_dir or ("results/" + ("smoke-" if args.smoke_test else "")
                                   + datetime.now().strftime("%Y-%m-%d-%H%M%S-%f")))
    checkpoint = run_dir / "models" / "checkpoint.pt"
    if args.skip_train:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint missing: {checkpoint}. Historical May outputs contain no saved models.")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "models").mkdir()
    figures_dir = Path(args.figures_dir) if args.figures_dir else run_dir / "plots"
    log_name = "evaluation_output.txt" if args.skip_train else "terminal_output.txt"
    with (run_dir / log_name).open("w") as log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        print(f"Run directory: {run_dir}")
        print("SYNTHETIC SMOKE TEST — not empirical findings" if args.smoke_test else "Empirical data run")
        data = synthetic_data(args.seed) if args.smoke_test else load_data(args.cache, args.allow_legacy_cache)
        fingerprint = data_fingerprint(data)
        splits = data["splits"]
        if args.skip_train:
            # Only load checkpoints you trust: this contains Python model objects.
            saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
            if saved["data_fingerprint"] != fingerprint:
                raise ValueError("Checkpoint data do not match the supplied cache/synthetic seed.")
            models = saved["models"]
            _move_models(models, args.device)
        else:
            try:
                revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
            except (OSError, subprocess.CalledProcessError):
                revision, dirty = None, None
            config = {**vars(args), "k": train.K_GRID, "lambda_lin": train.LAMBDA_LIN_GRID,
                      "lambda_nonlin": train.LAMBDA_NONLIN_GRID, "max_epochs": train.MAX_EPOCHS,
                      "patience": train.PATIENCE, "bootstrap_samples": n_bootstrap,
                      "seeds": seeds if args.multi_seed else [args.seed], "data_fingerprint": fingerprint,
                      "cache_version": data.get("cache_version", "legacy"), "git_revision": revision,
                      "git_dirty": dirty, "python": sys.version,
                      "versions": {name: importlib.metadata.version(name) for name in
                                   ("torch", "numpy", "pandas", "scipy", "scikit-learn", "yfinance")},
                      "splits": {name: {"start": str(s['returns'].index[0].date()),
                                          "end": str(s['returns'].index[-1].date()),
                                          "months": len(s['returns']), "stocks": s['returns'].shape[1]}
                                 for name, s in splits.items()}}
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            models = train.train_all_models(splits, k_list=train.K_GRID, device=args.device,
                                             multi_seed=args.multi_seed, seeds=seeds, run_dir=str(run_dir))
            torch.save({"models": models, "data_fingerprint": fingerprint}, checkpoint)
            _save_histories(models, run_dir / "loss_histories.json")
        stability = None
        evaluated = models
        if models.get("multi_seed"):
            from src.ensemble import build_ensembles, compute_seed_stability, validate_ensemble_improvement
            evaluated = build_ensembles(models)
            stability = compute_seed_stability(splits, models, str(run_dir))
            validate_ensemble_improvement(splits, models, evaluated, str(run_dir))
        summary, significance = run_evaluation(splits, evaluated, str(figures_dir), str(run_dir),
                                               stability_df=stability, n_bootstrap=n_bootstrap)
        evaluation_config = {
            "data_fingerprint": fingerprint, "cache_version": data.get("cache_version", "legacy"),
            "synthetic": bool(data.get("synthetic")), "device": args.device,
            "bootstrap_samples": n_bootstrap, "block_length": 6,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path("main.py"), *sorted(Path("src").glob("*.py"))]},
        }
        (run_dir / "evaluation_config.json").write_text(json.dumps(evaluation_config, indent=2) + "\n")
        story = format_paper_story(summary, significance)
        if data.get("synthetic"):
            story = "SYNTHETIC SMOKE TEST — not empirical findings\n\n" + story
        elif data.get("cache_version") == "legacy" or "cache_version" not in data:
            story = "LEGACY DATA DIAGNOSTIC — not a corrected replication\n\n" + story
        print("\n" + story)
        (run_dir / "results_summary.txt").write_text(story + "\n")


if __name__ == "__main__":
    main()
