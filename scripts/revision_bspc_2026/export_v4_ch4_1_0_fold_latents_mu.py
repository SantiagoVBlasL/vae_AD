#!/usr/bin/env python3
"""Export fold-wise VAE mu embeddings for the completed V4 [4,1,0] run.

This script is intentionally inference-only:
- it does not train or update any model;
- it does not modify the completed run directory;
- it uses saved outer-fold membership and VAE checkpoints;
- it writes only small CSV/JSON/README artefacts to the AUC sprint output tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"Missing src directory: {SRC_DIR}")

from betavae_xai.data.preprocessing import apply_normalization_params, normalize_inter_channel_fold
from betavae_xai.models import ConvolutionalVAE


DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/latent_exports_ch4_1_0"
DEFAULT_CONFIG_FALLBACK = PROJECT_ROOT / "configs/runs/adni_expanded_v4_beta25_ch4_1_0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-fallback", type=Path, default=DEFAULT_CONFIG_FALLBACK)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(run_dir: Path, fallback: Path) -> Dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_cfg = read_json(run_config_path)
        args = run_cfg.get("args", {})
        return {
            "source": str(run_config_path),
            "global_tensor_path": run_cfg.get("global_tensor_path") or args.get("global_tensor_path"),
            "metadata_path": run_cfg.get("metadata_path") or args.get("metadata_path"),
            "channels_to_use": run_cfg.get("channels_to_use_indices") or args.get("channels_to_use"),
            "selected_channel_names": run_cfg.get("channel_names_selected"),
            "latent_dim": args.get("latent_dim", 256),
            "dropout_rate_vae": args.get("dropout_rate_vae", 0.15),
            "vae_final_activation": args.get("vae_final_activation", "tanh"),
            "intermediate_fc_dim_vae": args.get("intermediate_fc_dim_vae", "quarter"),
            "use_layernorm_vae_fc": bool(args.get("use_layernorm_vae_fc", False)),
            "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
            "decoder_type": args.get("decoder_type", "convtranspose"),
            "norm_mode": args.get("norm_mode", "zscore_offdiag"),
        }

    fallback = resolve(fallback)
    cfg = read_json(fallback)
    params = cfg.get("parameters", {})
    paths = cfg.get("paths", {})
    return {
        "source": str(fallback),
        "global_tensor_path": paths.get("global_tensor_path"),
        "metadata_path": paths.get("metadata_path"),
        "channels_to_use": params.get("channels_to_use"),
        "selected_channel_names": cfg.get("selected_channel_names"),
        "latent_dim": params.get("latent_dim", 256),
        "dropout_rate_vae": params.get("dropout_rate_vae", 0.15),
        "vae_final_activation": params.get("vae_final_activation", "tanh"),
        "intermediate_fc_dim_vae": params.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(params.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(params.get("num_conv_layers_encoder", 4)),
        "decoder_type": params.get("decoder_type", "convtranspose"),
        "norm_mode": params.get("norm_mode", "zscore_offdiag"),
    }


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated_patterns = [
        "fold_*_trainDev_latents_mu.csv",
        "fold_*_test_latents_mu.csv",
        "fold_*_latent_subject_index.csv",
        "latent_export_manifest.json",
        "README.md",
    ]
    existing: List[Path] = []
    for pattern in generated_patterns:
        existing.extend(path.glob(pattern))
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains latent export files; pass --overwrite")
    if overwrite:
        for file_path in existing:
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
    return path


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artefacts:\n" + "\n".join(missing))


def load_selected_tensor(tensor_path: Path, channels: List[int]) -> Dict[str, Any]:
    with np.load(tensor_path, allow_pickle=True) as npz:
        if "global_tensor_data" not in npz.files:
            raise KeyError(f"{tensor_path} does not contain global_tensor_data")
        tensor = np.asarray(npz["global_tensor_data"][:, channels, :, :], dtype=np.float32)
        subject_ids = np.asarray(npz["subject_ids"]).astype(str) if "subject_ids" in npz.files else None
        channel_names = np.asarray(npz["channel_names"]).astype(str).tolist() if "channel_names" in npz.files else None
    return {"tensor": tensor, "subject_ids": subject_ids, "channel_names": channel_names}


def make_model(cfg: Dict[str, Any], image_size: int, n_channels: int, device: torch.device) -> ConvolutionalVAE:
    model = ConvolutionalVAE(
        input_channels=n_channels,
        latent_dim=int(cfg["latent_dim"]),
        image_size=image_size,
        final_activation=str(cfg["vae_final_activation"]),
        intermediate_fc_dim_config=cfg["intermediate_fc_dim_vae"],
        dropout_rate=float(cfg["dropout_rate_vae"]),
        use_layernorm_fc=bool(cfg["use_layernorm_vae_fc"]),
        num_conv_layers_encoder=int(cfg["num_conv_layers_encoder"]),
        decoder_type=str(cfg["decoder_type"]),
    )
    model.to(device)
    model.eval()
    return model


def encode_mu(model: ConvolutionalVAE, tensor: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    mus: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = torch.from_numpy(tensor[start : start + batch_size]).float().to(device)
            mu, _logvar = model.encode(batch)
            mus.append(mu.detach().cpu().numpy())
    return np.concatenate(mus, axis=0)


def subject_frame(
    subjects: pd.DataFrame,
    metadata: pd.DataFrame,
    mu: np.ndarray,
    fold: int,
    split: str,
) -> pd.DataFrame:
    base = subjects.copy()
    base["SubjectID"] = base["SubjectID"].astype(str)
    meta_cols = [c for c in ["SubjectID", "Age", "Sex"] if c in metadata.columns]
    if "SubjectID" not in meta_cols:
        raise ValueError("Metadata must contain SubjectID")
    base = base.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    base["y"] = base["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    base["fold"] = fold
    base["split"] = split
    latent_cols = [f"mu_{i:03d}" for i in range(mu.shape[1])]
    latent_df = pd.DataFrame(mu, columns=latent_cols)
    out = pd.concat([base.reset_index(drop=True), latent_df], axis=1)
    first_cols = ["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "y", "Age", "Sex", "fold", "split"]
    ordered = [c for c in first_cols if c in out.columns] + latent_cols
    return out[ordered]


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    cfg = load_config(run_dir, args.config_fallback)

    tensor_path = resolve(Path(str(cfg["global_tensor_path"])))
    metadata_path = resolve(Path(str(cfg["metadata_path"])))
    channels = [int(c) for c in cfg["channels_to_use"]]
    if channels != [4, 1, 0]:
        raise ValueError(f"This exporter is scoped to channels [4,1,0]; config requested {channels}")

    require_files([tensor_path, metadata_path, *[run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt" for fold in range(1, 6)]])
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    tensor_info = load_selected_tensor(tensor_path, channels)
    tensor = tensor_info["tensor"]
    metadata = pd.read_csv(metadata_path)
    selected_channel_names = cfg.get("selected_channel_names") or [
        tensor_info["channel_names"][i] for i in channels
    ]

    if tensor.ndim != 4 or tensor.shape[1] != len(channels):
        raise ValueError(f"Unexpected selected tensor shape: {tensor.shape}")

    if tensor_info["subject_ids"] is not None:
        tensor_subject_ids = pd.Series(tensor_info["subject_ids"]).astype(str)
        if len(tensor_subject_ids) != tensor.shape[0]:
            raise ValueError("subject_ids length does not match tensor N")

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()),
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "config_source": cfg["source"],
        "global_tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "channels_to_use": channels,
        "selected_channel_names": selected_channel_names,
        "latent_dim": int(cfg["latent_dim"]),
        "device": str(device),
        "no_retraining": True,
        "folds": [],
    }

    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        train_subjects_path = fold_dir / "train_dev_subjects_fold.csv"
        test_subjects_path = fold_dir / "test_subjects_fold.csv"
        vae_pool_idx_path = fold_dir / "vae_training_pool_tensor_idx.npy"
        vae_train_local_idx_path = fold_dir / "vae_actual_train_idx_local_to_pool.npy"
        checkpoint_path = fold_dir / f"vae_model_fold_{fold}.pt"
        require_files([train_subjects_path, test_subjects_path, vae_pool_idx_path, vae_train_local_idx_path, checkpoint_path])

        train_subjects = pd.read_csv(train_subjects_path)
        test_subjects = pd.read_csv(test_subjects_path)
        for split_name, subjects in [("trainDev", train_subjects), ("test", test_subjects)]:
            if subjects["tensor_idx"].duplicated().any():
                raise ValueError(f"Fold {fold} {split_name} has duplicate tensor_idx")
            idx = subjects["tensor_idx"].astype(int).to_numpy()
            if idx.min() < 0 or idx.max() >= tensor.shape[0]:
                raise IndexError(f"Fold {fold} {split_name} tensor_idx out of bounds")

        vae_pool_idx = np.load(vae_pool_idx_path).astype(int)
        vae_train_local_idx = np.load(vae_train_local_idx_path).astype(int)
        vae_pool_tensor = tensor[vae_pool_idx]
        _pool_norm, norm_params = normalize_inter_channel_fold(
            vae_pool_tensor,
            vae_train_local_idx,
            mode=str(cfg["norm_mode"]),
            selected_channel_original_names=list(selected_channel_names),
        )

        train_norm = apply_normalization_params(tensor[train_subjects["tensor_idx"].astype(int).to_numpy()], norm_params)
        test_norm = apply_normalization_params(tensor[test_subjects["tensor_idx"].astype(int).to_numpy()], norm_params)

        model = make_model(cfg, image_size=tensor.shape[-1], n_channels=tensor.shape[1], device=device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        mu_train = encode_mu(model, train_norm, args.batch_size, device)
        mu_test = encode_mu(model, test_norm, args.batch_size, device)

        train_out = subject_frame(train_subjects, metadata, mu_train, fold, "trainDev")
        test_out = subject_frame(test_subjects, metadata, mu_test, fold, "test")
        train_out.to_csv(outdir / f"fold_{fold}_trainDev_latents_mu.csv", index=False)
        test_out.to_csv(outdir / f"fold_{fold}_test_latents_mu.csv", index=False)

        index_cols = ["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "y", "Age", "Sex", "fold", "split"]
        subject_index = pd.concat([train_out[index_cols], test_out[index_cols]], ignore_index=True)
        subject_index["n_latent_dims"] = mu_train.shape[1]
        subject_index.to_csv(outdir / f"fold_{fold}_latent_subject_index.csv", index=False)

        manifest["folds"].append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint_path),
                "n_trainDev": int(train_out.shape[0]),
                "n_test": int(test_out.shape[0]),
                "latent_dim": int(mu_train.shape[1]),
                "normalization_recomputed_from_saved_vae_train_pool": True,
            }
        )

        del model, train_norm, test_norm, vae_pool_tensor, _pool_norm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (outdir / "latent_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = [
        "# V4 [4,1,0] Fold Latent Exports",
        "",
        "Read-only export from the completed beta=2.5 VAE checkpoints.",
        "",
        f"- Source run: `{run_dir}`",
        f"- Source realpath: `{run_dir.resolve()}`",
        f"- Tensor: `{tensor_path}`",
        f"- Metadata: `{metadata_path}`",
        f"- Channels: `{channels}` = {selected_channel_names}",
        f"- Latent dim: `{cfg['latent_dim']}`",
        "- No VAE training was run.",
        "- Fold normalization was recomputed from saved VAE training-pool indices and applied to train/dev and test splits.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(outdir), "folds_exported": len(manifest["folds"]), "no_retraining": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
