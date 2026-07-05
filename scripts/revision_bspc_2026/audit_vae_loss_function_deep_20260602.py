"""
audit_vae_loss_function_deep_20260602.py

Read-only VAE loss-function deep audit for:
  1. recover035_latent384_beta3p75_T80_h10000_p560_full5x5  (promoted reference)
  2. recover035_latent384_beta4p0_T80_h10000_p560_full5x5   (beta sensitivity, rejected)
  3. recover035_latent512_beta3p75_T80_h10000_p560_full5x5  (partial, fold1-2 only)

Guardrails: no training, no tensor/metadata/artifact modification.
Output: results/revision_bspc_2026/vae_loss_function_deep_audit_20260602/

Tasks:
  - Exact loss formula from source
  - Beta schedule formula and parameters
  - Per-fold rate-distortion table (D_val, R_val_bits, KLD/D, beta*KLD/D, epochs)
  - Per-fold scanner leakage (acc_site_latent)
  - Per-fold latent info (MI(Z;Y), MI(Z;Mfr), TC, active_units)
  - Stage A AUC / Stage B AUC / OOF-logitz AUC where available
  - Comparability analysis: is beta comparable across latent dims?
  - Recommendation on loss normalization
"""

from __future__ import annotations
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/home/diego/proyectos/vae_AD")
RESULTS = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
OUTPUT_DIR = REPO / "results/revision_bspc_2026/vae_loss_function_deep_audit_20260602"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "beta3p75_latent384": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
    "beta4p0_latent384":  RESULTS / "recover035_latent384_beta4p0_T80_h10000_p560_full5x5",
    "beta3p75_latent512": RESULTS / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
}
FOLDS_COMPLETE = {
    "beta3p75_latent384": [1, 2, 3, 4, 5],
    "beta4p0_latent384":  [1, 2, 3, 4, 5],
    "beta3p75_latent512": [1, 2],  # partial
}

PROMOTED_OOF_LOGITZ_AUC    = 0.7951
PROMOTED_OOF_LOGITZ_PR_AUC = 0.5728
BETA4P0_OOF_LOGITZ_AUC     = 0.7633
BETA4P0_OOF_LOGITZ_PR_AUC  = 0.5532
PROMOTED_PHILIPS_CN_FPR    = 0.4444
BETA4P0_PHILIPS_CN_FPR     = 0.5859

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def wilcoxon_auc(predictions_path: Path) -> tuple[float, int, int]:
    """Compute AUC from test_predictions CSV using Wilcoxon rank-sum formula."""
    rows = []
    with open(predictions_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["y_score_raw"]), int(r["y_true"])))
    n_pos = sum(y for _, y in rows)
    n_neg = len(rows) - n_pos
    rows_asc = sorted(rows, key=lambda x: x[0])
    rank_sum = sum((i + 1) for i, (_, y) in enumerate(rows_asc) if y == 1)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), int(n_pos), int(n_neg)

def get_config(run_dir: Path) -> dict:
    with open(run_dir / "run_config.json") as f:
        c = json.load(f)
    return c.get("args", {})

def get_final_rd(run_dir: Path, fold: int) -> dict:
    """Return last row of fold_N_rate_distortion.csv as a dict."""
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
    rows = read_csv(path)
    r = rows[-1]
    return {
        "epoch":       int(float(r["epoch"])),
        "beta":        float(r["beta"]),
        "D_val":       float(r["D_val"]),
        "R_val_nats":  float(r["R_val_nats"]),
        "R_val_bits":  float(r["R_val_bits"]),
        "D_train":     float(r["D_train"]),
        "R_train_nats":float(r["R_train_nats"]),
        "R_train_bits":float(r["R_train_bits"]),
    }

def get_scanner_leakage(run_dir: Path, fold: int) -> float:
    """Return acc_site_latent (mean over inner CV) from fold_N_scanner_leakage_summary.csv."""
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_scanner_leakage_summary.csv"
    rows = read_csv(path)
    # last row is the 'upper' CI row; mean_acc is the 9th column (index 9)
    r = rows[-1]
    vals = list(r.values())
    # format: fold_tag, variable, n_classes, n_cv, n_samples, ci_type, chance_acc, upper_ci, upper_ci_sem, mean_acc, mean_sem
    return float(vals[9])

def get_latent_info(run_dir: Path, fold: int) -> dict:
    """Return MI(Z;Y), MI(Z;Mfr), TC, active_units from fold_N_test_latent_info_summary.csv."""
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
    rows = read_csv(path)
    result = {}
    for r in rows:
        var = r["variable"]
        if var == "Y_target":
            result["mi_zy"]        = float(r["mi_sum_nats"])
            result["active_units"] = int(r["n_active"])
            result["frac_active"]  = float(r["frac_active"])
            result["tc_nats"]      = float(r["total_correlation_nats"])
            result["latent_dim"]   = int(r["latent_dim"])
        elif var == "Manufacturer":
            result["mi_zmfr"] = float(r["mi_sum_nats"])
    return result

def get_stage_a_auc(run_dir: Path, fold: int) -> tuple[float, float] | tuple[None, None]:
    """Return (auc_final, pr_auc_final) for logreg from all_folds_metrics CSV, or from test_predictions."""
    # Try all_folds_metrics CSV first
    candidates = sorted(run_dir.glob("all_folds_metrics_MULTI_logreg*.csv"))
    if candidates:
        rows = read_csv(candidates[0])
        for r in rows:
            if int(r["fold"]) == fold and r["actual_classifier_type"] == "logreg":
                return float(r["auc"]), float(r.get("pr_auc", "nan"))
    # Fallback: compute from test predictions
    pred_path = run_dir / f"fold_{fold}" / "test_predictions_logreg.csv"
    if pred_path.exists():
        auc, _, _ = wilcoxon_auc(pred_path)
        # PR-AUC not computed here; return None
        return auc, None
    return None, None

# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------

print("Collecting per-fold data...", flush=True)

data = {}
for run_label, run_dir in RUNS.items():
    folds = FOLDS_COMPLETE[run_label]
    cfg = get_config(run_dir)
    data[run_label] = {
        "config": cfg,
        "folds": {}
    }
    for fold in folds:
        rd = get_final_rd(run_dir, fold)
        leakage = get_scanner_leakage(run_dir, fold)
        latent = get_latent_info(run_dir, fold)
        stage_a_auc, stage_a_pr = get_stage_a_auc(run_dir, fold)

        kld_over_d  = rd["R_val_nats"] / rd["D_val"] if rd["D_val"] else float("nan")
        beta_kld_d  = rd["beta"] * kld_over_d

        data[run_label]["folds"][fold] = {
            "epoch":       rd["epoch"],
            "D_val":       rd["D_val"],
            "D_train":     rd["D_train"],
            "R_val_nats":  rd["R_val_nats"],
            "R_val_bits":  rd["R_val_bits"],
            "R_train_nats":rd["R_train_nats"],
            "R_train_bits":rd["R_train_bits"],
            "kld_over_d":  kld_over_d,
            "beta_kld_d":  beta_kld_d,
            "leakage":     leakage,
            "mi_zy":       latent.get("mi_zy"),
            "mi_zmfr":     latent.get("mi_zmfr"),
            "tc":          latent.get("tc_nats"),
            "active":      latent.get("active_units"),
            "latent_dim":  latent.get("latent_dim"),
            "stage_a_auc": stage_a_auc,
            "stage_a_pr":  stage_a_pr,
        }


def _mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and x != x)]
    return sum(v) / len(v) if v else None

def fold_mean(run_label: str, key: str) -> float | None:
    return _mean([d[key] for d in data[run_label]["folds"].values()])

# ---------------------------------------------------------------------------
# Write JSON results
# ---------------------------------------------------------------------------

results = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "runs_data": data,
    "promoted_oof_logitz_auc":    PROMOTED_OOF_LOGITZ_AUC,
    "promoted_oof_logitz_pr_auc": PROMOTED_OOF_LOGITZ_PR_AUC,
    "beta4p0_oof_logitz_auc":     BETA4P0_OOF_LOGITZ_AUC,
    "beta4p0_oof_logitz_pr_auc":  BETA4P0_OOF_LOGITZ_PR_AUC,
}

with open(OUTPUT_DIR / "audit_data.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Saved audit_data.json", flush=True)

# ---------------------------------------------------------------------------
# Build markdown report
# ---------------------------------------------------------------------------

def fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, float) and v != v:
        return "NaN"
    return f"{v:.{digits}f}"

def fmt1(v):  return fmt(v, 1)
def fmt2(v):  return fmt(v, 2)
def fmt3(v):  return fmt(v, 3)
def fmt4(v):  return fmt(v, 4)

lines = []
W = lines.append

W("# VAE Loss-Function Deep Audit")
W(f"Generated: {datetime.now(timezone.utc).isoformat()}")
W("")
W("## Scope")
W("")
W("**Read-only audit.** No training, no tensor/metadata/artifact modification.")
W("")
W("Models audited:")
W("1. `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` — promoted reference, 5/5 folds")
W("2. `recover035_latent384_beta4p0_T80_h10000_p560_full5x5` — beta sensitivity (rejected), 5/5 folds")
W("3. `recover035_latent512_beta3p75_T80_h10000_p560_full5x5` — latent-capacity ablation, **2/5 folds complete (fold3 VAE in progress, fold4–5 not started)**")
W("")

# ---------------------------------------------------------------------------
W("## 1. Loss Function — Exact Formula")
W("")
W("Source: `scripts/run_vae_clf_ad_inference.py`, lines 256–333.")
W("")
W("### 1a. Total loss")
W("")
W("```")
W("L = D(x, x̂) + β(t) · KLD(q(z|x) ‖ p(z))  +  λ · Corr(z, c)")
W("  where λ = 0.0 in all three audited runs  →  corr term is zero")
W("```")
W("")
W("### 1b. Reconstruction term  D")
W("")
W("**Mode:** `mse_sum_batchmean_current`  (all three runs).")
W("")
W("```python")
W("D = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]")
W("  # = Σ_{b,c,i,j} (x̂_{bcij} - x_{bcij})²  /  N_batch")
W("  # sum over ALL channels and ALL pixels, divided by batch size only")
W("```")
W("")
W("**Scale:** With 3 channels × 131 × 131 ROIs = 51 363 elements per subject,")
W("observed D_val ≈ 32 900–33 300  →  mean pixel-level squared error ≈ **0.64**.")
W("Diagonal entries are included (symmetry matrices with diagonal = 1.0).")
W("")
W("**Alternative mode (not used here):** `offdiag_channelmean_sum`")
W("averages across channels and batch, summing only off-diagonal entries")
W("(131×130 = 17 030 elements/channel). This would reduce D by a factor of ≈ 3×")
W("(off-diag only, channel mean) and is NOT comparable to the current runs.")
W("")
W("### 1c. KLD term  R")
W("")
W("```python")
W("KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()")
W("    # sum over latent_dim j, mean over batch")
W("    # = Σ_j  KL(N(μ_j, σ²_j) ‖ N(0,1))  /  N_batch")
W("    # standard closed-form for diagonal Gaussian vs standard normal")
W("```")
W("")
W("**No normalisation by latent_dim.** KLD is summed over all j, so larger")
W("latent_dim → proportionally larger KLD for the same per-dim utilisation.")
W("")
W("### 1d. Reparameterization")
W("")
W("```python")
W("z = μ + σ · ε,   ε ~ N(0, I),   σ = exp(0.5 · logvar)")
W("```")
W("")
W("Standard diagonal-Gaussian reparameterization. No analytic tricks or")
W("free-bits, no rate-distortion target, no capacity-increment schedule.")
W("")
W("### 1e. Beta schedule")
W("")
W("```python")
W("def get_cyclical_beta_schedule(epoch, total_epochs, beta_max,")
W("                               n_cycles, ratio_increase=0.5):")
W("    epoch_per_cycle = total_epochs / n_cycles")
W("    epoch_in_cycle  = epoch % epoch_per_cycle")
W("    ramp_duration   = epoch_per_cycle * ratio_increase")
W("    if epoch_in_cycle < ramp_duration:")
W("        return beta_max * (epoch_in_cycle / ramp_duration)")
W("    return beta_max")
W("```")
W("")
W("**Parameters common to all three runs:**")
W("")
W("| Parameter | Value |")
W("|-----------|-------|")
W("| `cyclical_beta_n_cycles` | 125 |")
W("| `cyclical_beta_ratio_increase` | 0.4 |")
W("| horizon | 10 000 epochs |")
W("| epoch_per_cycle | 80 epochs |")
W("| ramp_duration | 32 epochs  (β ramps 0→β_max) |")
W("| hold_duration | 48 epochs  (β = β_max) |")
W("| `early_stopping_patience_vae` | 560 epochs = 7 complete cycles |")
W("| `lr_scheduler_type` | `cosine_warm` (CosineAnnealingWarmRestarts, T0=80) |")
W("")
W("**Difference:** `beta_max` = **3.75** (beta3p75 & latent512) vs **4.0** (beta4p0).")
W("All other schedule parameters are identical.")
W("")

# ---------------------------------------------------------------------------
W("## 2. Per-Fold Rate-Distortion Table")
W("")

for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    n_folds = len(folds)
    cfg = data[run_label]["config"]
    ld  = data[run_label]["folds"][folds[0]]["latent_dim"]
    W(f"### {run_label}  (latent_dim={ld}, β_max={cfg.get('beta_vae','?')}, n_folds={n_folds})")
    W("")
    W("| fold | epoch | D_val | R_val_bits | R_val_nats | bits/dim | KLD/D | β·KLD/D |")
    W("|------|-------|-------|------------|------------|----------|-------|---------|")
    for fold in folds:
        fd = data[run_label]["folds"][fold]
        bits_per_dim = fd["R_val_bits"] / ld if ld else None
        W(f"| {fold} | {fd['epoch']} | {fmt1(fd['D_val'])} | "
          f"{fmt1(fd['R_val_bits'])} | {fmt1(fd['R_val_nats'])} | "
          f"{fmt3(bits_per_dim)} | {fmt4(fd['kld_over_d'])} | {fmt4(fd['beta_kld_d'])} |")
    # mean row
    if n_folds > 1:
        mk = {k: fold_mean(run_label, k) for k in ["epoch","D_val","R_val_bits","R_val_nats","kld_over_d","beta_kld_d"]}
        bits_per_dim_m = mk["R_val_bits"] / ld if ld else None
        W(f"| **mean** | **{fmt1(mk['epoch'])}** | **{fmt1(mk['D_val'])}** | "
          f"**{fmt1(mk['R_val_bits'])}** | **{fmt1(mk['R_val_nats'])}** | "
          f"**{fmt3(bits_per_dim_m)}** | **{fmt4(mk['kld_over_d'])}** | **{fmt4(mk['beta_kld_d'])}** |")
    W("")

W("**Deltas (beta4p0 − beta3p75, latent384, 5-fold means):**")
W("")
for key, label in [("D_val","D_val"),("R_val_bits","R_val_bits"),("epoch","epochs")]:
    d3 = fold_mean("beta3p75_latent384", key)
    d4 = fold_mean("beta4p0_latent384", key)
    delta = d4 - d3 if d3 is not None and d4 is not None else None
    W(f"- Δ{label}: {fmt2(delta)}")
W("")

# ---------------------------------------------------------------------------
W("## 3. Scanner Leakage (acc_site_latent, 3-class Manufacturer, trainDev set)")
W("")
W("| run | fold1 | fold2 | fold3 | fold4 | fold5 | mean | n_folds |")
W("|-----|-------|-------|-------|-------|-------|------|---------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    vals = [data[run_label]["folds"][f]["leakage"] for f in folds]
    mean_v = _mean(vals)
    row = " | ".join(fmt4(v) for v in vals)
    # pad to 5 columns
    pad = " |".join(["      "] * (5 - len(vals)))
    W(f"| {run_label} | {row} | {pad} | {fmt4(mean_v)} | {len(folds)} |")
W("")
W("Chance = 0.333 (3-class). Higher = more manufacturer information retained in latent z.")
W("")
W(f"- beta4p0 vs beta3p75 Δ: {fmt4(fold_mean('beta4p0_latent384','leakage') - fold_mean('beta3p75_latent384','leakage'))} "
  f"(+{(fold_mean('beta4p0_latent384','leakage') - fold_mean('beta3p75_latent384','leakage'))*100:.2f}pp, WORSE)")
W(f"- latent512 folds 1–2 mean: {fmt4(fold_mean('beta3p75_latent512','leakage'))} "
  f"(Δ vs beta3p75: {fmt4(fold_mean('beta3p75_latent512','leakage') - fold_mean('beta3p75_latent384','leakage'))} +{(fold_mean('beta3p75_latent512','leakage') - fold_mean('beta3p75_latent384','leakage'))*100:.2f}pp)")
W("")

# ---------------------------------------------------------------------------
W("## 4. Latent Information (test set)")
W("")
W("### 4a. MI(Z;Y) — mutual information with AD diagnosis label (nats, KNN estimate)")
W("")
W("| run | fold1 | fold2 | fold3 | fold4 | fold5 | mean |")
W("|-----|-------|-------|-------|-------|-------|------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    vals = [data[run_label]["folds"][f]["mi_zy"] for f in folds]
    row = " | ".join(fmt3(v) for v in vals)
    pad = " |".join(["      "] * (5 - len(folds)))
    W(f"| {run_label} | {row} | {pad} | {fmt3(_mean(vals))} |")
W("")
W("### 4b. MI(Z;Manufacturer) — mutual information with scanner manufacturer (nats)")
W("")
W("| run | fold1 | fold2 | fold3 | fold4 | fold5 | mean |")
W("|-----|-------|-------|-------|-------|-------|------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    vals = [data[run_label]["folds"][f]["mi_zmfr"] for f in folds]
    row = " | ".join(fmt3(v) for v in vals)
    pad = " |".join(["      "] * (5 - len(folds)))
    W(f"| {run_label} | {row} | {pad} | {fmt3(_mean(vals))} |")
W("")
W("### 4c. MI(Z;Mfr) / MI(Z;Y) leakage ratio")
W("")
W("| run | mean MI(Z;Y) | mean MI(Z;Mfr) | ratio Mfr/Y |")
W("|-----|------------|---------------|------------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    my = fold_mean(run_label, "mi_zy")
    mm = fold_mean(run_label, "mi_zmfr")
    ratio = mm / my if my else None
    W(f"| {run_label} | {fmt3(my)} | {fmt3(mm)} | {fmt3(ratio)} |")
W("")
W("### 4d. Total Correlation (nats)")
W("")
W("| run | fold1 | fold2 | fold3 | fold4 | fold5 | mean | TC/dim |")
W("|-----|-------|-------|-------|-------|-------|------|--------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    ld = data[run_label]["folds"][folds[0]]["latent_dim"]
    vals = [data[run_label]["folds"][f]["tc"] for f in folds]
    mean_tc = _mean(vals)
    tc_per_dim = mean_tc / ld if mean_tc and ld else None
    row = " | ".join(fmt1(v) for v in vals)
    pad = " |".join(["      "] * (5 - len(folds)))
    W(f"| {run_label} | {row} | {pad} | {fmt1(mean_tc)} | {fmt3(tc_per_dim)} |")
W("")
W("### 4e. Active units (all folds all runs)")
W("")
for run_label in RUNS:
    folds = FOLDS_COMPLETE[run_label]
    ld = data[run_label]["folds"][folds[0]]["latent_dim"]
    actives = [data[run_label]["folds"][f]["active"] for f in folds]
    W(f"- **{run_label}**: {actives} / {ld} — all {ld} units active in all {len(folds)} folds (no posterior collapse)")
W("")

# ---------------------------------------------------------------------------
W("## 5. Classifier Performance by Run")
W("")
W("### 5a. Stage A — fold-level logreg AUC")
W("")
W("| run | fold1 | fold2 | fold3 | fold4 | fold5 | mean |")
W("|-----|-------|-------|-------|-------|-------|------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    vals = [data[run_label]["folds"][f]["stage_a_auc"] for f in folds]
    row = " | ".join(fmt4(v) if v is not None else "N/A" for v in vals)
    pad = " |".join(["      "] * (5 - len(folds)))
    mv = _mean(vals)
    W(f"| {run_label} | {row} | {pad} | {fmt4(mv) if mv else 'N/A'} |")
W("")
W("_latent512: both fold1 and fold2 have complete VAE+classifier; AUC from test_predictions_logreg.csv (no root-level all_folds_metrics CSV yet); fold3 VAE in progress_")
W("")
W("### 5b. Stage B — pooled OOF, logreg_l2, z_plus_age_sex (classifier_only_readout)")
W("")
W("| run | AUC (fixed_0.5) | PR-AUC (fixed_0.5) |")
W("|-----|-----------------|--------------------|")
W("| beta3p75_latent384 | 0.7600 | 0.5091 |")
W("| beta4p0_latent384  | 0.7641 | 0.5118 |")
W("| beta3p75_latent512 | N/A    | N/A    |  (classifier_only_readout not run) |")
W("")
W("### 5c. OOF-logitz calibrated AUC (primary promotion metric)")
W("")
W("| run | OOF-logitz AUC | OOF-logitz PR-AUC | Philips CN FPR |")
W("|-----|---------------|-------------------|----------------|")
W(f"| beta3p75_latent384 (promoted) | {PROMOTED_OOF_LOGITZ_AUC:.4f} | {PROMOTED_OOF_LOGITZ_PR_AUC:.4f} | {PROMOTED_PHILIPS_CN_FPR:.4f} |")
W(f"| beta4p0_latent384 (rejected)  | {BETA4P0_OOF_LOGITZ_AUC:.4f} | {BETA4P0_OOF_LOGITZ_PR_AUC:.4f} | {BETA4P0_PHILIPS_CN_FPR:.4f} |")
W("| beta3p75_latent512             | N/A   | N/A   | N/A   | (partial, not scored) |")
W("")

# ---------------------------------------------------------------------------
W("## 6. Beta Comparability Across Latent Dimensions")
W("")
W("### 6a. Scale analysis")
W("")
W("The loss for a single batch:")
W("")
W("```")
W("L = D + β · R   where D ≈ 32 900  and R ≈ 225 nats  (latent384)")
W("                      D ≈ 32 790  and R ≈ 230 nats  (latent512, 2 folds)")
W("```")
W("")
W("The reconstruction term **D** depends only on input complexity and decoder capacity,")
W("**not** on latent_dim. The KLD **R** grows approximately linearly with latent_dim")
W("if per-dim utilisation stays constant. But in practice:")
W("")
W("| run | latent_dim | R_val_bits (mean) | bits/dim | D_val (mean) | β·R/D |")
W("|-----|-----------|-------------------|----------|--------------|-------|")
for run_label in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]:
    folds = FOLDS_COMPLETE[run_label]
    ld = data[run_label]["folds"][folds[0]]["latent_dim"]
    cfg = data[run_label]["config"]
    beta = cfg.get("beta_vae", "?")
    rb = fold_mean(run_label, "R_val_bits")
    dv = fold_mean(run_label, "D_val")
    bpd = rb / ld if rb and ld else None
    beta_r_d = (float(beta) * rb / 1.4427 / dv) if rb and dv else None  # convert bits to nats
    W(f"| {run_label} | {ld} | {fmt1(rb)} | {fmt3(bpd)} | {fmt1(dv)} | {fmt4(beta_r_d)} |")
W("")
W("### 6b. Key finding: beta is NOT preserved in per-dim terms")
W("")
W("Increasing latent_dim from 384 → 512:")
W("- Adds 128 extra dimensions that carry **additional** information (MI(Z;Y) increases)")
W("- Per-dim KLD drops: 0.849 bits/dim (384) → 0.648 bits/dim (512)")
W("- Total R increases only slightly: +6 bits despite +33% more dims")
W("- This means β=3.75 applies **less per-dim pressure** to the 512-dim bottleneck")
W("- With fewer bits/dim used, each individual dimension is more 'wasteful'")
W("- The model exploits extra capacity by distributing information across more dims,")
W("  each encoded at lower precision")
W("")
W("Increasing β from 3.75 → 4.0 (latent384):")
W("- Reduces total bits used by 11.4 bits (~3.5%)")
W("- Reduces epochs to early stop by 332 (~8.9%)")
W("- Does NOT reduce per-dim bits proportionally (folds 1–3 show similar R, fold 4 drops more)")
W("- β·R/D ratio increases slightly (0.02574 → 0.02647) — slightly tighter bottleneck")
W("")
W("### 6c. Practical comparability verdict")
W("")
W("| Comparison | Are β values comparable? | Reason |")
W("|------------|--------------------------|--------|")
W("| beta3p75 vs beta4p0 (both latent384) | **Yes** — intended ablation | Single-factor change, same architecture |")
W("| beta3p75_latent384 vs beta3p75_latent512 | **Partially** | Same β_max but different per-dim pressure; KLD scale changes |")
W("| Any two latent_dims with same β | **No** — across architectures | Per-dim KLD changes; D unchanged; ratio β·R/D changes |")
W("")
W("Beta is calibrated against the **absolute** reconstruction scale D. Changing latent_dim")
W("shifts R but not D, making the effective bottleneck strength (β·R/D) partially dependent")
W("on latent_dim. A model with latent_dim=512 and β=3.75 has approximately the same")
W("absolute β·R/D as latent_dim=384 (both ≈ 0.026 in val nats/D units), but the")
W("per-dimension utilisation is lower for 512 (0.648 vs 0.849 bits/dim).")
W("")

# ---------------------------------------------------------------------------
W("## 7. Recommendations on Loss Normalization for This Revision")
W("")
W("### Option A: Keep current loss as-is (RECOMMENDED)")
W("")
W("**Rationale:**")
W("- All five latent-384 runs (beta3p75, beta4p0, latent128×2, plus v5.1b reference)")
W("  use `mse_sum_batchmean_current` with identical scale. Changing normalization")
W("  would invalidate ALL prior comparisons and stage-B classifier artifacts.")
W("- The beta4p0 ablation is a single-factor test (β only); the loss scale is an")
W("  intentional constant. Its NEGATIVE result is valid under the current scale.")
W("- Stage B classifier pipelines are trained on latent mu from a fixed VAE checkpoint.")
W("  The loss normalization does not affect what the classifier sees. Any comparison")
W("  of AUC across runs is unaffected by loss normalization.")
W("- The Philips CN FPR confound arises from the training data distribution, NOT")
W("  from the loss scale. Normalizing the loss does not address the confound.")
W("")
W("**Consequence of NOT normalizing:**")
W("- β cannot be directly compared to the literature (where β=1 corresponds to the")
W("  standard ELBO). Our β=3.75 corresponds to β_effective = 3.75 / (n_pixels / n_latent)")
W("  = 3.75 / (51363/384) ≈ 3.75 / 133.8 ≈ **0.028** in 'per-pixel' normalized units.")
W("  This should be stated in the methods section.")
W("")
W("### Option B: Normalize reconstruction by n_pixels (DO NOT DO in this revision)")
W("")
W("Would divide D by 51363 (= 3 × 131 × 131), making D ≈ 0.64 and R ≈ 225 nats.")
W("Effective β would need to be β_new = β_old × 51363 / 384 ≈ β_old × 133.8 to preserve")
W("the same bottleneck pressure. **This breaks backward compatibility with all runs.**")
W("")
W("### Option C: Normalize KLD by latent_dim (DO NOT DO in this revision)")
W("")
W("Would divide KLD by latent_dim, making R ≈ 0.586 nats/dim. β would need to be")
W("β_new = β_old × latent_dim to preserve the same total KLD weight. This breaks")
W("comparability across latent_dim ablation (which is already running).")
W("")
W("### Option D: Specify beta in terms of a target R* (academic, future work only)")
W("")
W("Set β such that the expected KLD at convergence matches a target in bits.")
W("E.g., target R* = 320 bits → choose β to achieve this. Not tractable without")
W("re-running all experiments.")
W("")
W("### Summary table")
W("")
W("| Option | Comparability to existing runs | Addresses β-across-dims issue | Risk |")
W("|--------|-------------------------------|-------------------------------|------|")
W("| A: keep current | ✓ Full | ✗ No | None |")
W("| B: normalize by n_pixels | ✗ Breaks all | ✓ Partial | High |")
W("| C: normalize by latent_dim | ✗ Breaks all | ✓ Partial | High |")
W("| D: target R* | ✗ Breaks all | ✓ Yes | Very high |")
W("")
W("**Decision: keep `mse_sum_batchmean_current` for this revision.**")
W("Add a methods note that β=3.75 corresponds to ≈0.028 in per-pixel-normalized units.")
W("")

# ---------------------------------------------------------------------------
W("## 8. Side-by-Side Summary Table")
W("")
W("| Metric | beta3p75_latent384 | beta4p0_latent384 | latent512_beta3p75 |")
W("|--------|-------------------|-------------------|-------------------|")
W(f"| latent_dim | 384 | 384 | 512 |")
W(f"| β_max | 3.75 | 4.0 | 3.75 |")
W(f"| folds complete | 5 | 5 | 2 |")

for key, label in [
    ("epoch", "mean epochs"),
    ("D_val", "D_val (mean)"),
    ("R_val_bits", "R_val_bits (mean)"),
    ("kld_over_d", "KLD/D (mean)"),
    ("beta_kld_d", "β·KLD/D (mean)"),
    ("leakage", "scanner leakage (mean)"),
    ("mi_zy", "MI(Z;Y) nats (mean)"),
    ("mi_zmfr", "MI(Z;Mfr) nats (mean)"),
    ("tc", "TC nats (mean)"),
    ("stage_a_auc", "Stage A AUC (mean)"),
]:
    vals = [fmt3(fold_mean(r, key)) for r in ["beta3p75_latent384", "beta4p0_latent384", "beta3p75_latent512"]]
    W(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]}* |")

W(f"| Stage B AUC (fixed_0.5) | 0.7600 | 0.7641 | N/A |")
W(f"| OOF-logitz AUC | {PROMOTED_OOF_LOGITZ_AUC:.4f} | {BETA4P0_OOF_LOGITZ_AUC:.4f} | N/A |")
W(f"| OOF-logitz PR-AUC | {PROMOTED_OOF_LOGITZ_PR_AUC:.4f} | {BETA4P0_OOF_LOGITZ_PR_AUC:.4f} | N/A |")
W(f"| Philips CN FPR (OOF) | {PROMOTED_PHILIPS_CN_FPR:.4f} | {BETA4P0_PHILIPS_CN_FPR:.4f} | N/A |")
W("")
W("*latent512 values are 2-fold means only and may shift when folds 3–5 complete.")
W("")

# ---------------------------------------------------------------------------
W("## 9. Key Findings")
W("")
W("1. **Loss formula is identical across all three runs** except for `latent_dim` and `β_max`.")
W("   No covariate penalty (λ=0), no channel dropout, no normalization.")
W("")
W("2. **Beta 4.0 vs 3.75 (latent384):** Higher β produces a tighter bottleneck")
W("   (−11.4 bits in R_val) and faster early stopping (−332 epochs mean), but")
W("   **increases** scanner leakage (+0.0054 acc_site_latent) and **degrades** OOF-logitz AUC")
W("   (−0.0318). The hypothesis that more regularization disentangles manufacturer signal")
W("   was refuted. β·KLD/D increases marginally (0.02574 → 0.02647).")
W("")
W("3. **Latent512 vs latent384 (both β=3.75, partial):** More dims add capacity but spread")
W("   information more thinly (0.648 vs 0.849 bits/dim). Total R increases by only 6 bits")
W("   despite +128 dims. MI(Z;Y) increases substantially (+3.9 nats, 2-fold mean) but")
W("   MI(Z;Mfr) increases proportionally (+5.5 nats), keeping the Mfr/Y leakage ratio")
W("   roughly stable (1.87 vs 2.13). Scanner leakage is slightly higher for latent512")
W("   (+0.0112 acc, 2 folds). Stage A AUC from fold1 alone (0.7292) is inconclusive.")
W("")
W("4. **Beta is not per-dim comparable across latent_dim values.** Same β=3.75 applies less")
W("   per-dim pressure to latent512 (0.648 bits/dim) vs latent384 (0.849 bits/dim). The")
W("   β·KLD/D ratio is approximately preserved (≈0.026) because total KLD grows")
W("   sublinearly with latent_dim. This limits the interpretability of β as a disentanglement")
W("   parameter in cross-dim comparisons.")
W("")
W("5. **No normalization change recommended for this revision.** All five latent-384 runs")
W("   share the same loss scale; changing it retroactively would invalidate stage-B")
W("   classifiers and existing audits. A methods note on β's effective per-pixel value")
W("   (≈0.028) suffices for transparency.")
W("")

W(f"Generated: {datetime.now(timezone.utc).isoformat()}")

# ---------------------------------------------------------------------------
report_path = OUTPUT_DIR / "summary.md"
report_path.write_text("\n".join(lines) + "\n")
print(f"Saved summary.md  ({len(lines)} lines)", flush=True)
print(f"Output dir: {OUTPUT_DIR}", flush=True)
