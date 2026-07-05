#!/usr/bin/env python
"""Update manuscript defense package with Manufacturer-conditioned completeness audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFENSE_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
)
AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "conditional_beta_vae_manufacturer_full5x5_completeness_audit"
)
CANDIDATE = "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"


COMPLETENESS_DECISION = (
    "not promoted as primary and no longer-horizon rerun justified: completeness audit PASS for data integrity; "
    "0/5 Manufacturer-conditioned folds reached max epoch, 0/5 had best epoch in the last 10% of the 4480 budget, "
    "all folds early-stopped with epochs_after_best=320; 035_S_6927 was present only in the VAE pool, absent from "
    "supervised folds, 128_S_2002 was absent, classifier pool remained n=396/CN=300/AD=96; logreg_l2 selected "
    "C=0.001 in 4/5 folds but prior ultra-regularized readout was negative and OASIS advantage was unstable across "
    "concatenated, runwise164, and runwise140TR handling; retain as secondary deconfounding/sensitivity analysis"
)


SECTION = """## Manufacturer-Conditioned FULL 5x5 Completeness Audit

Audit path: `results/revision_bspc_2026/conditional_beta_vae_manufacturer_full5x5_completeness_audit/`

Decision: `no_longer_rerun_justified`.

The read-only training/data/hyperparameter completeness audit found no evidence that the Manufacturer-conditioned FULL 5x5 run was undertrained or right-censored at the 4480-epoch horizon:

- Manufacturer-conditioned folds reaching max epoch: `0/5`.
- Best epoch in last 10% of the 4480-epoch budget: `0/5`.
- All folds early-stopped with `epochs_after_best=320`.
- Data integrity passed: `035_S_6927` was present in the VAE pool only, absent from supervised train/test, `128_S_2002` was absent, and the classifier pool remained `n=396`, `CN=300`, `AD=96`.
- `logreg_l2` selected `C=0.001` in `4/5` folds. This does not justify expanding the classifier grid because the prior ultra-regularized readout audit was negative.
- The OASIS external advantage was not stable across run handling: concatenated timeseries improved AUC/PR-AUC, runwise164 was mixed, and runwise140TR had lower AUC.

Decision: do not run a longer-horizon Manufacturer-conditioned rerun, do not promote this branch as the primary model, and retain it only as a secondary deconfounding/sensitivity analysis.
"""


REVIEWER_SECTION = """## Reviewer Response: Manufacturer-Conditioned Completeness Audit

We also performed a read-only training/data/hyperparameter completeness audit of the corrected Manufacturer-conditioned FULL 5x5 branch. This audit asked whether the conditional branch might have been unfairly limited by the 4480-epoch training horizon or by incomplete data/hyperparameter selection.

The answer was negative. None of the Manufacturer-conditioned folds reached the maximum epoch (`0/5`), none selected its best checkpoint in the last 10% of the training budget (`0/5`), and all folds early-stopped with `epochs_after_best=320`. Data integrity also passed: `035_S_6927` was restored only to the unsupervised VAE metadata, excluded from all supervised classifier folds, `128_S_2002` remained absent, and the classifier pool stayed fixed at `n=396`, `CN=300`, `AD=96`.

Although `logreg_l2` selected the lower grid value `C=0.001` in `4/5` folds, a prior ultra-regularized readout audit showed that stronger regularization hurt test ranking. In addition, the OASIS external advantage of Manufacturer conditioning was not stable across run handling: concatenated timeseries improved AUC/PR-AUC, runwise164 was mixed, and the ADNI-like runwise140TR tensor showed lower AUC. Therefore, a longer-horizon Manufacturer-conditioned rerun is not scientifically justified. We retain this branch as a secondary deconfounding/sensitivity audit rather than changing the primary manuscript model.
"""


def replace_section(text: str, header: str, replacement: str) -> str:
    if header not in text:
        return text.rstrip() + "\n\n" + replacement.rstrip() + "\n"
    start = text.index(header)
    next_start = text.find("\n## ", start + 1)
    if next_start == -1:
        return text[:start].rstrip() + "\n\n" + replacement.rstrip() + "\n"
    return text[:start].rstrip() + "\n\n" + replacement.rstrip() + "\n\n" + text[next_start + 1 :].lstrip()


def write_failed_table() -> None:
    csv_path = DEFENSE_DIR / "failed_optimization_table.csv"
    df = pd.read_csv(csv_path)
    mask = df["candidate"].eq(CANDIDATE)
    if not mask.any():
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "candidate": CANDIDATE,
                            "category": "conditional beta-VAE Manufacturer conditioning",
                            "evaluation_stage": "FULL 5x5 secondary deconfounding/completeness audit",
                            "decision": COMPLETENESS_DECISION,
                            "auc": 0.77125,
                            "pr_auc": 0.590093,
                            "balanced_accuracy": 0.703958,
                            "sensitivity": 0.697917,
                            "specificity": 0.71,
                            "f1": 0.536,
                            "delta_auc_vs_locked": -0.011701,
                            "delta_pr_auc_vs_locked": 0.03022,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        df.loc[mask, "evaluation_stage"] = "FULL 5x5 secondary deconfounding/completeness audit"
        df.loc[mask, "decision"] = COMPLETENESS_DECISION
    df.to_csv(csv_path, index=False)
    md_path = DEFENSE_DIR / "failed_optimization_table.md"
    md_path.write_text(
        "# Failed / Non-Promoted Optimization Table\n\n" + df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def update_markdown(path: Path, section: str, header: str) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(replace_section(text, header, section), encoding="utf-8")


def update_command_log() -> None:
    path = DEFENSE_DIR / "command_log.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    updates = payload.setdefault("updates", [])
    action = "add_manufacturer_conditioned_full5x5_completeness_audit"
    updates = [u for u in updates if u.get("action") != action]
    updates.append(
        {
            "action": action,
            "source_dir": str(AUDIT_DIR),
            "decision": "no_longer_rerun_justified",
            "candidate": CANDIDATE,
            "recorded_findings": {
                "manufacturer_conditioned_folds_reaching_max_epoch": "0/5",
                "best_epoch_in_last_10pct_of_4480_budget": "0/5",
                "epochs_after_best_all_folds": 320,
                "035_S_6927_present_in_vae_pool_only": True,
                "035_S_6927_absent_from_supervised_train_test": True,
                "128_S_2002_absent": True,
                "classifier_pool": {"n": 396, "cn": 300, "ad": 96},
                "logreg_l2_lower_bound_C_0p001_hits": "4/5",
                "prior_ultra_regularized_readout_negative": True,
                "oasis_advantage_stable_across_run_handling": False,
            },
            "final_decision": "do_not_rerun_longer_horizon_do_not_promote_primary_keep_secondary_deconfounding_sensitivity",
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "configs_modified": False,
            "existing_model_outputs_modified": False,
            "updated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    payload["updates"] = updates
    payload["last_updated_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload["latest_update_script"] = str(Path(__file__).resolve())
    payload["training_launched"] = False
    payload["tensor_modified"] = False
    payload["metadata_modified"] = False
    payload["ledger_modified"] = False
    payload["existing_results_modified"] = False
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if not AUDIT_DIR.exists():
        raise FileNotFoundError(AUDIT_DIR)
    write_failed_table()
    update_markdown(DEFENSE_DIR / "final_recommendation.md", SECTION, "## Manufacturer-Conditioned FULL 5x5 Completeness Audit")
    update_markdown(DEFENSE_DIR / "README.md", SECTION, "## Manufacturer-Conditioned FULL 5x5 Completeness Audit")
    update_markdown(DEFENSE_DIR / "conditional_beta_vae_manufacturer_full5x5_result.md", SECTION, "## Manufacturer-Conditioned FULL 5x5 Completeness Audit")
    update_markdown(DEFENSE_DIR / "reviewer_response_ready_text.md", REVIEWER_SECTION, "## Reviewer Response: Manufacturer-Conditioned Completeness Audit")
    update_command_log()
    print(json.dumps({"updated": str(DEFENSE_DIR), "decision": "no_longer_rerun_justified"}, indent=2))


if __name__ == "__main__":
    main()
