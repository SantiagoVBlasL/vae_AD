#!/usr/bin/env python3
"""Read-only [1,2] channel-pair feasibility and beta-scaling audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
OUT = ROOT / "channel12_beta_scaling_feasibility_audit_20260605"

PROMOTED = ROOT / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
CH1ONLY = ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5"
OLD_CH12 = ROOT / "adni_v5_1b_ch1_2_offdiag_channelmean_horizon4480_cycles56_full_5x5"

PROMOTED_CFG = Path("configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json")
CH1ONLY_CFG = Path("configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.json")
OLD_CH12_CFG = Path("configs/runs/adni_v5_1b_ch1_2_offdiag_channelmean_horizon4480_cycles56_full_5x5.json")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def params(path: Path) -> dict[str, Any]:
    cfg = read_json(path)
    p = dict(cfg.get("parameters", {}))
    p["selected_channel_names"] = cfg.get("selected_channel_names")
    p["metadata_path"] = cfg.get("paths", {}).get("metadata_path")
    p["global_tensor_path"] = cfg.get("paths", {}).get("global_tensor_path")
    p["output_dir"] = cfg.get("paths", {}).get("output_dir")
    p["big_disk_output_dir"] = cfg.get("paths", {}).get("big_disk_output_dir")
    p["run_name"] = cfg.get("run_name")
    p["description"] = cfg.get("description")
    return p


def write_table(stem: str, df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{stem}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    (OUT / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def discover_ch12_configs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg_path in sorted(Path("configs/runs").glob("*.json")):
        try:
            p = params(cfg_path)
        except Exception:
            continue
        channels = p.get("channels_to_use")
        selected = p.get("selected_channel_names")
        is_ch12 = channels == [1, 2] or selected == ["Pearson_Full_FisherZ_Signed", "MI_KNN_Symmetric"]
        if not is_ch12:
            continue
        output_dir = p.get("output_dir")
        rows.append(
            {
                "config_path": str(cfg_path),
                "run_name": p.get("run_name"),
                "output_dir": output_dir,
                "output_exists": Path(output_dir).exists() if output_dir else False,
                "channels_to_use": json.dumps(channels),
                "selected_channel_names": json.dumps(selected),
                "latent_dim": p.get("latent_dim"),
                "beta_vae": p.get("beta_vae"),
                "epochs_vae": p.get("epochs_vae"),
                "cyclical_beta_n_cycles": p.get("cyclical_beta_n_cycles"),
                "lr_scheduler_T0": p.get("lr_scheduler_T0"),
                "early_stopping_patience_vae": p.get("early_stopping_patience_vae"),
                "recon_loss_mode": p.get("recon_loss_mode"),
                "metadata_path": p.get("metadata_path"),
            }
        )
    return pd.DataFrame(rows)


def matching_requirements(configs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    wanted = {
        "channels_to_use": "[1, 2]",
        "latent_dim": 384,
        "beta_vae": 3.75,
        "epochs_vae": 10000,
        "cyclical_beta_n_cycles": 125,
        "lr_scheduler_T0": 80,
        "early_stopping_patience_vae": 560,
        "recon_loss_mode": "mse_sum_batchmean_current",
    }
    for _, r in configs.iterrows():
        checks = {
            "recover035_metadata": "patched_metadata_candidate.csv" in str(r.get("metadata_path", "")),
            "channels_1_2": r.get("channels_to_use") == wanted["channels_to_use"],
            "latent_dim_384": r.get("latent_dim") == wanted["latent_dim"],
            "beta_3p75": r.get("beta_vae") == wanted["beta_vae"],
            "T80_h10000": r.get("epochs_vae") == wanted["epochs_vae"] and r.get("lr_scheduler_T0") == wanted["lr_scheduler_T0"],
            "cycles_125": r.get("cyclical_beta_n_cycles") == wanted["cyclical_beta_n_cycles"],
            "patience_560": r.get("early_stopping_patience_vae") == wanted["early_stopping_patience_vae"],
            "full5x5_name": "full_5x5" in str(r.get("run_name", "")) or "full5x5" in str(r.get("run_name", "")),
            "mse_current": r.get("recon_loss_mode") == wanted["recon_loss_mode"],
        }
        rows.append(
            {
                "config_path": r.get("config_path"),
                "run_name": r.get("run_name"),
                **checks,
                "matches_all_requested": all(checks.values()),
            }
        )
    return pd.DataFrame(rows)


def rd_for_run(label: str, run_dir: Path, cfg_path: Path) -> pd.DataFrame:
    p = params(cfg_path)
    beta = float(p.get("beta_vae"))
    latent_dim = int(p.get("latent_dim"))
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not path.exists():
            rows.append({"run": label, "fold": fold, "status": "missing_rate_distortion"})
            continue
        df = pd.read_csv(path)
        best = df.loc[df["L_val_betaMax"].idxmin()]
        final = df.iloc[-1]
        rows.append(
            {
                "run": label,
                "fold": fold,
                "status": "available",
                "latent_dim": latent_dim,
                "beta_vae": beta,
                "best_epoch": int(best["epoch"]),
                "final_epoch": int(final["epoch"]),
                "D_val": float(best["D_val"]),
                "R_val_nats": float(best["R_val_nats"]),
                "R_val_bits": float(best["R_val_bits"]),
                "R_bits_per_latent_dim": float(best["R_val_bits"] / latent_dim),
                "R_over_D": float(best["R_val_nats"] / best["D_val"]),
                "beta_KLD_over_D": float(beta * best["R_val_nats"] / best["D_val"]),
                "L_val_betaMax": float(best["L_val_betaMax"]),
            }
        )
    return pd.DataFrame(rows)


def latent_signal_for_run(label: str, run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        li = run_dir / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        leak = run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
        row: dict[str, Any] = {"run": label, "fold": fold}
        if li.exists():
            df = pd.read_csv(li)
            by_var = {str(r["variable"]): r for _, r in df.iterrows()}
            y = by_var.get("Y_target")
            mfr = by_var.get("Manufacturer")
            if y is not None:
                row["MI_Z_Y_nats"] = float(y["mi_sum_nats"])
                row["active_units"] = int(y["n_active"])
                row["total_correlation_nats"] = float(y["total_correlation_nats"])
            if mfr is not None:
                row["MI_Z_Manufacturer_nats"] = float(mfr["mi_sum_nats"])
            if y is not None and mfr is not None and float(y["mi_sum_nats"]) != 0:
                row["MI_Manufacturer_over_MI_Y"] = float(mfr["mi_sum_nats"] / y["mi_sum_nats"])
        if leak.exists():
            s = pd.read_csv(leak).iloc[0]
            row["test_scanner_raw_ba"] = float(s["acc_site_raw"])
            row["test_scanner_latent_ba"] = float(s["acc_site_latent"])
            row["test_scanner_latent_minus_raw"] = row["test_scanner_latent_ba"] - row["test_scanner_raw_ba"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, g in df.groupby(group_col, dropna=False):
        row = {group_col: label}
        for metric in metrics:
            if metric in g.columns:
                row[f"{metric}_mean"] = pd.to_numeric(g[metric], errors="coerce").mean()
                row[f"{metric}_std"] = pd.to_numeric(g[metric], errors="coerce").std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def beta_estimates(rd: pd.DataFrame) -> pd.DataFrame:
    means = summarize(rd, "run", ["R_over_D", "beta_KLD_over_D", "D_val", "R_val_bits", "R_bits_per_latent_dim"])
    promoted = means[means["run"] == "promoted_ch1_0_2"].iloc[0]
    ch1 = means[means["run"] == "ch1only"].iloc[0]
    proxy = means[means["run"] == "existing_v5_1b_ch1_2_offdiag"].iloc[0]
    proxy_r_over_d = float(proxy["R_over_D_mean"])
    targets = [
        ("match_promoted_ch1_0_2_beta_KLD_over_D", float(promoted["beta_KLD_over_D_mean"])),
        ("match_ch1only_beta_KLD_over_D", float(ch1["beta_KLD_over_D_mean"])),
    ]
    rows: list[dict[str, Any]] = []
    for target_name, target_ratio in targets:
        rows.append(
            {
                "estimate_target": target_name,
                "target_beta_KLD_over_D": target_ratio,
                "proxy_source": "existing_v5_1b_ch1_2_offdiag_channelmean_latent256_beta2p5",
                "proxy_R_over_D_mean": proxy_r_over_d,
                "beta_needed_if_proxy_R_over_D_holds": target_ratio / proxy_r_over_d if proxy_r_over_d else np.nan,
                "caveat": "Proxy run differs in metadata, latent_dim, horizon/patience, and recon_loss_mode; use as rough scaling only.",
            }
        )
    rows.extend(
        [
            {
                "estimate_target": "conservative_same_beta_candidate",
                "target_beta_KLD_over_D": np.nan,
                "proxy_source": "promoted_model_control",
                "proxy_R_over_D_mean": np.nan,
                "beta_needed_if_proxy_R_over_D_holds": 3.75,
                "caveat": "Best for isolating channel subset only; does not force a rate-distortion match.",
            },
            {
                "estimate_target": "rounded_ch1only_rate_distortion_candidate",
                "target_beta_KLD_over_D": float(ch1["beta_KLD_over_D_mean"]),
                "proxy_source": "existing_ch12_proxy_rounded",
                "proxy_R_over_D_mean": proxy_r_over_d,
                "beta_needed_if_proxy_R_over_D_holds": 4.0,
                "caveat": "Rounded from proxy estimate; beta4.0/4.25 would confound channel-pair test with beta-axis change and beta4p0 was negative in latent384.",
            },
        ]
    )
    return pd.DataFrame(rows)


def write_recommendation(configs: pd.DataFrame, match: pd.DataFrame, rd_summary: pd.DataFrame, beta: pd.DataFrame) -> None:
    exact_exists = bool(match.get("matches_all_requested", pd.Series(dtype=bool)).any())
    old_ch12 = rd_summary[rd_summary["run"] == "existing_v5_1b_ch1_2_offdiag"].iloc[0]
    promoted = rd_summary[rd_summary["run"] == "promoted_ch1_0_2"].iloc[0]
    ch1 = rd_summary[rd_summary["run"] == "ch1only"].iloc[0]
    beta_promoted = beta[beta["estimate_target"] == "match_promoted_ch1_0_2_beta_KLD_over_D"].iloc[0]
    beta_ch1 = beta[beta["estimate_target"] == "match_ch1only_beta_KLD_over_D"].iloc[0]
    lines = [
        "# Final Recommendation",
        "",
        "Decision: **do_not_run_[1,2]_as_next_FULL**.",
        "",
        "Rationale:",
        f"- Exact requested [1,2] recover035 latent384 beta3.75/T80/h10000/p560/mse-current run exists: **{exact_exists}**.",
        "- The only completed FULL [1,2] run found is an older v5.1b latent256 beta2.5 horizon4480/offdiag-channelmean branch. It is useful historical evidence but not a direct match.",
        "- That older [1,2] branch was already a negative FULL confirmation: it did not beat the final [1,0,2] model or the simplified [1] model.",
        "- The current ch1-only latent384 beta3.75 branch improved internal AUC/PR-AUC but failed the OASIS comparison against promoted [1,0,2]; adding MI to Pearson Full has not shown robust benefit in prior FAST/FULL evidence.",
        "- The final model-selection synthesis already concluded no additional internal FULL tuning is justified without a stronger pre-specified reason.",
        "",
        "Rate-distortion context:",
        f"- Promoted [1,0,2] mean beta*KLD/D: {promoted['beta_KLD_over_D_mean']:.6f}.",
        f"- ch1-only mean beta*KLD/D: {ch1['beta_KLD_over_D_mean']:.6f}.",
        f"- Existing nonmatching [1,2] proxy mean beta*KLD/D: {old_ch12['beta_KLD_over_D_mean']:.6f}.",
        "",
        "Beta guidance if a reviewer-driven [1,2] FULL sensitivity is still required:",
        "- **Conservative same-beta candidate:** beta=3.75. This isolates the channel-pair change and is the cleaner scientific test.",
        f"- **Rate-distortion matched candidate to ch1-only:** beta≈{beta_ch1['beta_needed_if_proxy_R_over_D_holds']:.2f}, rounded to beta=4.0, but this is based on a nonmatching proxy and would confound channel-pair with beta-axis tuning.",
        f"- **Rate-distortion matched candidate to promoted [1,0,2]:** beta≈{beta_promoted['beta_needed_if_proxy_R_over_D_holds']:.2f}; this is not recommended because it would substantially reduce beta relative to the promoted branch and is inconsistent with the beta-axis evidence.",
        "",
        "Recommended action:",
        "- Do not launch a [1,2] FULL run now.",
        "- If a final reviewer-facing sensitivity is mandatory, run **[1,2] beta3.75** only, with a clear pre-specified negative/sensitivity framing.",
        "- Do not run beta4.0/beta4.25 as the first [1,2] test; beta4.0 was already negative in latent384 and would make interpretation less clean.",
    ]
    (OUT / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# Channel [1,2] Beta-Scaling Feasibility Audit",
        "",
        f"Generated: {now_iso()}",
        "",
        "Read-only audit for a possible Pearson Full + MI-KNN `[1,2]` latent384 FULL branch.",
        "",
        "No training, tensor modification, metadata modification, or model artifact modification was performed.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    command_log = {
        "timestamp_start": now_iso(),
        "argv": sys.argv,
        "guardrails": ["no training", "no tensor modification", "no metadata modification", "no model artifact modification"],
    }
    if args.dry_run:
        command_log["dry_run"] = True
        command_log["timestamp_end"] = now_iso()
        (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n")
        print("dry_run_ok")
        return 0

    configs = discover_ch12_configs()
    match = matching_requirements(configs)
    rd = pd.concat(
        [
            rd_for_run("promoted_ch1_0_2", PROMOTED, PROMOTED_CFG),
            rd_for_run("ch1only", CH1ONLY, CH1ONLY_CFG),
            rd_for_run("existing_v5_1b_ch1_2_offdiag", OLD_CH12, OLD_CH12_CFG),
        ],
        ignore_index=True,
    )
    rd_summary = summarize(rd, "run", ["D_val", "R_val_bits", "R_bits_per_latent_dim", "R_over_D", "beta_KLD_over_D"])
    signal = pd.concat(
        [
            latent_signal_for_run("promoted_ch1_0_2", PROMOTED),
            latent_signal_for_run("ch1only", CH1ONLY),
            latent_signal_for_run("existing_v5_1b_ch1_2_offdiag", OLD_CH12),
        ],
        ignore_index=True,
    )
    signal_summary = summarize(
        signal,
        "run",
        [
            "MI_Z_Y_nats",
            "MI_Z_Manufacturer_nats",
            "MI_Manufacturer_over_MI_Y",
            "active_units",
            "total_correlation_nats",
            "test_scanner_raw_ba",
            "test_scanner_latent_ba",
            "test_scanner_latent_minus_raw",
        ],
    )
    beta = beta_estimates(rd)

    write_table("existing_ch12_run_inventory", configs)
    write_table("matching_ch12_requirements", match)
    write_table("reference_rate_distortion_foldwise", rd)
    write_table("reference_rate_distortion_summary", rd_summary)
    write_table("reference_latent_nuisance_signal_foldwise", signal)
    write_table("reference_latent_nuisance_signal_summary", signal_summary)
    write_table("beta_scaling_estimates", beta)
    write_recommendation(configs, match, rd_summary, beta)

    command_log["timestamp_end"] = now_iso()
    command_log["dry_run"] = False
    command_log["outputs"] = sorted(p.name for p in OUT.iterdir())
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"output_dir={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
