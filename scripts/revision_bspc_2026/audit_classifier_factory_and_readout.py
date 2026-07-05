#!/usr/bin/env python3
"""Static audit of classifier factory and classifier-only ADNI v5.1 readout.

This script writes documentation artifacts only. It does not train classifiers,
does not touch VAE checkpoints, and does not read or modify tensor data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/revision_bspc_2026/classifier_factory_and_readout_audit"
CLASSIFIERS_PY = ROOT / "src/betavae_xai/models/classifiers.py"
SWEEP_SCRIPT = ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
THRESHOLD_AUDIT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_threshold_final_audit"
SWEEP_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def canonical_factory_rows() -> List[Dict[str, object]]:
    return [
        {
            "source": "classifiers.py",
            "classifier_key": "logreg",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "factory_available": True,
            "pipeline_preprocessing": "_AutoPreprocessor; scales numeric for non-tree models; Sex encoded M=0/F=1 then imputed/scaled",
            "imbalance_handling": "class_weight='balanced' if balance=True; optional SMOTE inside imblearn Pipeline",
            "calibration": "calibrate=True only emits deprecation warning; post-hoc calibration expected outside factory",
            "probability_or_score": "predict_proba available; canonical original run used calibrated y_score_final",
            "hyperparameter_ranges": "C: FloatDistribution(1e-5, 1, log=True); implicit penalty=l2; solver=liblinear; max_iter=20000",
            "suggested_n_iter": 900,
            "audit_assessment": "Conceptually compatible with classifier-only logreg_l2, but needs explicit no-calibration readout mode and reproducible preprocessing choice.",
        },
        {
            "source": "classifiers.py",
            "classifier_key": "svm",
            "estimator": "sklearn.svm.SVC",
            "factory_available": True,
            "pipeline_preprocessing": "_AutoPreprocessor; scales numeric; optional SMOTE/feature selection",
            "imbalance_handling": "class_weight='balanced' if balance=True; optional SMOTE",
            "calibration": "Factory model has probability=False; original pipeline can calibrate post-hoc outside factory",
            "probability_or_score": "decision_function unless post-hoc calibrated",
            "hyperparameter_ranges": "C: FloatDistribution(1e-1, 1e4, log=True); gamma: FloatDistribution(1e-7, 1e-1, log=True); kernel='rbf'",
            "suggested_n_iter": 900,
            "audit_assessment": "Canonical search is very broad for this N; refined readout should use tighter C/gamma around observed useful region.",
        },
        {
            "source": "classifiers.py",
            "classifier_key": "gb",
            "estimator": "lightgbm.LGBMClassifier",
            "factory_available": True,
            "pipeline_preprocessing": "_AutoPreprocessor without scaling unless SMOTE is active",
            "imbalance_handling": "class_weight='balanced' if balance=True; optional SMOTE",
            "calibration": "No factory calibration; calibrate flag deprecated",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": (
                "max_depth 3-12; num_leaves 4-64; bagging_fraction 0.5-1.0; feature_fraction 0.5-1.0; "
                "bagging_freq 1-10; learning_rate 5e-4-0.01 log; n_estimators 300-1000; "
                "min_child_samples 5-50; min_child_weight 1e-3-10 log; min_split_gain 0-1; "
                "reg_alpha/reg_lambda 1e-3-1 log"
            ),
            "suggested_n_iter": 180,
            "audit_assessment": "Learning-rate range is likely too low for the successful shallow classifier-only regime; keep as secondary only.",
        },
        {
            "source": "classifiers.py",
            "classifier_key": "rf",
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "factory_available": True,
            "pipeline_preprocessing": "_AutoPreprocessor without scaling unless SMOTE is active",
            "imbalance_handling": "class_weight='balanced' if balance=True; optional SMOTE",
            "calibration": "No factory calibration; calibrate flag deprecated",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "n_estimators 100-1200; max_features sqrt/log2/0.2/0.4; max_depth 8-50; min_samples_split 2-30; min_samples_leaf 1-20",
            "suggested_n_iter": 150,
            "audit_assessment": "Good balanced-accuracy operating point in the sweep, but lower AUC/PR-AUC and more complex; not recommended as primary readout.",
        },
        {
            "source": "classifiers.py",
            "classifier_key": "mlp",
            "estimator": "sklearn.neural_network.MLPClassifier",
            "factory_available": True,
            "pipeline_preprocessing": "_AutoPreprocessor with scaling",
            "imbalance_handling": "No class_weight in sklearn MLP; optional SMOTE",
            "calibration": "No factory calibration; calibrate flag deprecated",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "hidden_layer_sizes from config default 128,64; alpha 1e-5-1e-1 log; learning_rate_init 1e-5-1e-2 log; max_iter=1000; early_stopping=True",
            "suggested_n_iter": 200,
            "audit_assessment": "Not part of the minimal refined sweep; small clinical sample makes MLP a higher-variance readout.",
        },
        {
            "source": "classifiers.py",
            "classifier_key": "xgb",
            "estimator": "xgboost.XGBClassifier",
            "factory_available": True,
            "pipeline_preprocessing": "_AutoPreprocessor without scaling unless SMOTE is active",
            "imbalance_handling": "Factory does not map balance=True to scale_pos_weight; optional SMOTE only",
            "calibration": "No factory calibration; calibrate flag deprecated",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "gamma 0-5; n_estimators 500-1500; learning_rate 1e-4-0.1 log; max_depth 2-8; subsample 0.3-1; colsample_bytree 0.5-1; min_child_weight 0.5-10 log",
            "suggested_n_iter": 200,
            "audit_assessment": "Secondary only; factory imbalance handling should set scale_pos_weight per train fold if used.",
        },
    ]


def classifier_only_rows() -> List[Dict[str, object]]:
    return [
        {
            "source": "classifier-only sweep script",
            "classifier_key": "logreg_l2",
            "estimator": "LogisticRegression(penalty='l2', solver='lbfgs')",
            "factory_available": False,
            "pipeline_preprocessing": "ColumnTransformer: z-scales mu, median-imputes/scales Age, one-hot encodes Sex",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "No calibration",
            "probability_or_score": "predict_proba; median score substantially higher than original calibrated logreg",
            "hyperparameter_ranges": "C grid [0.001, 0.01, 0.1, 1.0]",
            "suggested_n_iter": "grid=4",
            "audit_assessment": "Primary readout candidate; all folds selected C=0.001, so next grid should expand below and around 0.001.",
        },
        {
            "source": "classifier-only sweep script",
            "classifier_key": "logreg_elasticnet",
            "estimator": "LogisticRegression(penalty='elasticnet', solver='saga')",
            "factory_available": False,
            "pipeline_preprocessing": "Same classifier-only ColumnTransformer",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "No calibration",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "C grid [0.01, 0.1, 1.0]; l1_ratio [0.2, 0.7]",
            "suggested_n_iter": "grid=6",
            "audit_assessment": "Useful sparse sensitivity check; l1_ratio grid is too coarse and C should focus around 0.01-0.3.",
        },
        {
            "source": "classifier-only sweep script",
            "classifier_key": "svm_rbf",
            "estimator": "SVC(kernel='rbf', probability=True)",
            "factory_available": False,
            "pipeline_preprocessing": "Same classifier-only ColumnTransformer",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "SVC probability=True Platt scaling inside training fold; no outer-test threshold selection",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "C grid [0.1, 1, 10]; gamma ['scale', 0.001]",
            "suggested_n_iter": "grid=6",
            "audit_assessment": "Current compact grid is narrow and best C often hits 0.1; expand modestly below 0.1.",
        },
        {
            "source": "classifier-only sweep script",
            "classifier_key": "random_forest",
            "estimator": "RandomForestClassifier(n_estimators=300)",
            "factory_available": False,
            "pipeline_preprocessing": "Same classifier-only ColumnTransformer",
            "imbalance_handling": "class_weight='balanced_subsample'",
            "calibration": "No calibration",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "max_depth [4, None]; min_samples_leaf [2, 8]",
            "suggested_n_iter": "grid=4",
            "audit_assessment": "Not in minimal refined sweep despite good BA; AUC/PR-AUC lag LogReg L2.",
        },
        {
            "source": "classifier-only sweep script",
            "classifier_key": "gradient_boosting/lightgbm/xgboost",
            "estimator": "GradientBoostingClassifier, LGBMClassifier, XGBClassifier",
            "factory_available": "partial",
            "pipeline_preprocessing": "Same classifier-only ColumnTransformer",
            "imbalance_handling": "LGBM class_weight balanced; XGB scale_pos_weight; sklearn GB none",
            "calibration": "No calibration",
            "probability_or_score": "predict_proba",
            "hyperparameter_ranges": "shallow grids: n_estimators 100/250; learning_rate 0.03/0.1; max_depth 2 or num_leaves 7",
            "suggested_n_iter": "compact grid",
            "audit_assessment": "Secondary only; helpful robustness check, not current primary readout.",
        },
    ]


def refined_plan_rows() -> List[Dict[str, object]]:
    threshold_rules = "fixed_0p5 plus true-inner-CV-OOF Youden, balanced_accuracy, target_sens>=0.70 max specificity"
    return [
        {
            "priority": 1,
            "classifier": "logreg_l2",
            "include": "yes_primary",
            "estimator": "LogisticRegression(penalty='l2', solver='lbfgs' or 'liblinear')",
            "proposed_grid": "C [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "off for primary readout; optional calibration sensitivity analysis only",
            "threshold_rules": threshold_rules,
            "reason": "Best AUC/PR-AUC and primary threshold result; current grid selected lower boundary C=0.001 in every fold.",
            "risk_control": "Keep same outer folds and latent cache; no VAE retraining.",
        },
        {
            "priority": 2,
            "classifier": "logreg_elasticnet",
            "include": "yes_sensitivity",
            "estimator": "LogisticRegression(penalty='elasticnet', solver='saga')",
            "proposed_grid": "C [0.003, 0.01, 0.03, 0.1, 0.3]; l1_ratio [0.05, 0.1, 0.2, 0.5, 0.8]",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "off for comparability to logreg_l2",
            "threshold_rules": threshold_rules,
            "reason": "Tests sparse/regularized readout stability without expanding model class too far.",
            "risk_control": "Report as secondary if AUC/PR-AUC or subgroup behavior trails L2.",
        },
        {
            "priority": 3,
            "classifier": "svm_rbf",
            "include": "yes_sensitivity",
            "estimator": "SVC(kernel='rbf', probability=True) or decision_function with inner-OOF thresholding",
            "proposed_grid": "C [0.03, 0.1, 0.3, 1, 3]; gamma [1e-4, 3e-4, 1e-3, 3e-3, 'scale']",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "avoid extra calibration unless nested inside train/dev only",
            "threshold_rules": threshold_rules,
            "reason": "Current compact grid often selected C=0.1 and gamma=0.001; canonical grid is too broad for a minimal readout sweep.",
            "risk_control": "Treat as robustness comparator; do not select threshold on outer tests.",
        },
        {
            "priority": 4,
            "classifier": "lightgbm",
            "include": "optional_secondary",
            "estimator": "LGBMClassifier(objective='binary')",
            "proposed_grid": "n_estimators [100, 250, 400]; learning_rate [0.01, 0.03, 0.1]; num_leaves [3, 7, 15]; min_child_samples [10, 25, 50]; reg_lambda [0.1, 1, 10]",
            "imbalance_handling": "class_weight='balanced'",
            "calibration": "off; optional post-hoc only inside train/dev",
            "threshold_rules": threshold_rules,
            "reason": "Secondary nonlinear check; canonical LR range is likely too low for this readout.",
            "risk_control": "Do not promote over LogReg L2 unless AUC/PR-AUC and subgroup metrics improve.",
        },
        {
            "priority": 5,
            "classifier": "xgboost",
            "include": "optional_secondary",
            "estimator": "XGBClassifier(objective='binary:logistic')",
            "proposed_grid": "n_estimators [100, 250, 500]; learning_rate [0.01, 0.03, 0.1]; max_depth [1, 2, 3]; min_child_weight [1, 5, 10]; reg_lambda [1, 10]",
            "imbalance_handling": "scale_pos_weight = n_neg/n_pos per outer train fold",
            "calibration": "off; optional post-hoc only inside train/dev",
            "threshold_rules": threshold_rules,
            "reason": "Secondary nonlinear check; factory balance=True does not currently set scale_pos_weight.",
            "risk_control": "Keep shallow to reduce overfit; report as secondary only.",
        },
    ]


def load_context() -> Dict[str, object]:
    comparison = pd.read_csv(require(SWEEP_DIR / "comparison_vs_original_logreg_svm.csv"))
    prob = pd.read_csv(require(THRESHOLD_AUDIT_DIR / "original_vs_classifier_only_probability_distribution.csv"))
    threshold_log = json.loads(require(THRESHOLD_AUDIT_DIR / "command_log.json").read_text(encoding="utf-8"))
    status = pd.read_csv(require(SWEEP_DIR / "classifier_sweep_model_status.csv"))

    def row(model: str, strategy: str) -> Dict[str, object]:
        match = comparison[
            (comparison["model_name"] == model)
            & (comparison["threshold_strategy"] == strategy)
            & (comparison["row_type"] == "absolute")
        ]
        if match.empty:
            return {}
        return match.iloc[0].to_dict()

    prob_summary = prob[prob["class_label"] == "ALL"].set_index("model_name").to_dict(orient="index")
    return {
        "original_logreg": row("original_logreg", "original_fixed_0p5"),
        "original_svm": row("original_svm", "original_fixed_0p5"),
        "logreg_l2_0p5": row("logreg_l2", "fixed_0p5"),
        "logreg_l2_target": row("logreg_l2", "inner_oof_target_sens_ge_0p70_max_spec"),
        "random_forest_youden": row("random_forest", "inner_oof_youden_j"),
        "probability_summary": prob_summary,
        "threshold_log": threshold_log,
        "logreg_l2_status": status[status["model_name"] == "logreg_l2"].to_dict(orient="records"),
    }


def write_readme(context: Dict[str, object]) -> None:
    orig = context["original_logreg"]
    svm = context["original_svm"]
    l2 = context["logreg_l2_0p5"]
    target = context["logreg_l2_target"]
    rf = context["random_forest_youden"]
    prob = context["probability_summary"]
    orig_median = prob.get("original_logreg", {}).get("score_median", float("nan"))
    l2_median = prob.get("logreg_l2", {}).get("score_median", float("nan"))
    threshold_log = context["threshold_log"]

    readme = f"""# Classifier Factory and Readout Audit

## Scope

Inputs reviewed:

- `src/betavae_xai/models/classifiers.py`
- `scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py`
- `results/revision_bspc_2026/adni_v5_1_batch20260514b_threshold_final_audit`
- `results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep`

No VAE training, tensor modification, metadata modification, or ledger modification was performed.

## Available canonical classifiers

`classifiers.py` exposes `rf`, `gb`, `svm`, `logreg`, `mlp`, and `xgb`. The detailed search spaces are in `classifier_factory_summary.csv`.

## Original pipeline vs classifier-only LogReg L2

- Original LogReg: L2 logistic regression, `solver=liblinear`, Optuna C range `1e-5..1`, class weighting enabled, post-hoc calibration enabled.
- Classifier-only `logreg_l2`: L2 logistic regression, `solver=lbfgs`, C grid `[0.001, 0.01, 0.1, 1.0]`, `class_weight=balanced`, no calibration.
- Original SVM: RBF SVC, broad C/gamma search, class weighting enabled, calibrated output used for threshold 0.5 reporting.
- Classifier-only SVM: compact RBF grid with `probability=True`.

Probability scale changed materially: original LogReg median score was `{orig_median:.3f}` versus classifier-only LogReg L2 median score `{l2_median:.3f}`. This explains why threshold 0.5 is conservative in the original run but much less conservative in the classifier-only readout.

At threshold 0.5, original LogReg sensitivity/specificity was `{orig.get('sensitivity', float('nan')):.3f}/{orig.get('specificity', float('nan')):.3f}` and original SVM was `{svm.get('sensitivity', float('nan')):.3f}/{svm.get('specificity', float('nan')):.3f}`. Classifier-only LogReg L2 at 0.5 moved to `{l2.get('sensitivity', float('nan')):.3f}/{l2.get('specificity', float('nan')):.3f}`. With the leakage-safe target-sensitivity rule it moved to `{target.get('sensitivity', float('nan')):.3f}/{target.get('specificity', float('nan')):.3f}`.

## Integration feasibility

Classifier-only `logreg_l2` can be integrated cleanly into the canonical classifier layer without changing VAE or tensor logic. It consumes latent features plus Age/Sex metadata downstream of the VAE. The integration should be explicit, however: either add a `logreg_l2` alias/readout mode to the factory or add factory parameters for penalty, solver, calibration policy, and threshold policy. Do not bury this under the existing `logreg` key without recording the readout semantics.

The current canonical `logreg` is close but not identical: it uses `liblinear`, `_AutoPreprocessor` Sex encoding/scaling, and the broader Optuna C range; the classifier-only sweep uses one-hot Sex, `lbfgs`, compact grid, and no calibration.

## Hyperparameter range audit

- Classifier-only `logreg_l2` is too narrow at the lower boundary: every fold selected `C=0.001`. Expand below and around that value.
- Canonical SVM is too broad for a minimal readout sweep (`C` up to `1e4`, `gamma` up to `1e-1`). Use a focused grid around observed useful values first.
- Canonical LightGBM learning rates (`5e-4..0.01`) are likely too low for the shallow successful classifier-only regime; keep LightGBM secondary with a practical shallow grid.
- Canonical XGBoost does not currently translate `balance=True` to `scale_pos_weight`; fix that before using XGBoost as a serious secondary readout.
- Random forest had good thresholded balanced accuracy (`{rf.get('balanced_accuracy', float('nan')):.3f}`) but lower AUC/PR-AUC than LogReg L2, so it should not drive the next minimal search.

## Threshold requirement

All non-0.5 operating points must use true inner-CV out-of-fold predictions from train/dev only. The existing threshold audit reports `threshold_selection_verified={threshold_log.get('threshold_selection_verified')}` and `outer_test_threshold_leakage={threshold_log.get('outer_test_threshold_leakage')}`.

## Recommendation

Run one minimal refined classifier-only sweep on saved latents:

1. `logreg_l2` as the primary readout.
2. `logreg_elasticnet` as a sparse stability check.
3. `svm_rbf` as a nonlinear margin comparator.
4. LightGBM/XGBoost only as secondary, shallow nonlinear checks.

Keep the same outer folds, same latent cache, same metadata, and require fixed 0.5 plus true-inner-OOF threshold rules for every classifier.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_code_notes() -> None:
    notes = """# Code Risk Notes

## Main risks

1. `logreg_l2` is not a named canonical factory key yet.
   - The canonical `logreg` is L2 logistic regression, but the classifier-only readout differs in solver, preprocessing, calibration, C grid, and threshold policy.
   - Add an explicit readout alias or config block instead of silently changing `logreg`.

2. Calibration semantics are split across layers.
   - `classifiers.py` warns that `calibrate=True` is deprecated and does not wrap the model.
   - The original pipeline still uses calibrated `y_score_final`.
   - Classifier-only `logreg_l2` is intentionally uncalibrated, which changes score scale and threshold behavior.

3. Preprocessing semantics differ.
   - Factory `_AutoPreprocessor` encodes Sex as numeric M=0/F=1 and scales it.
   - Classifier-only sweep one-hot encodes Sex and separately scales latent dimensions and Age.
   - Reproducing paper tables requires freezing this choice.

4. XGBoost imbalance handling is incomplete in the factory.
   - `balance=True` does not set `scale_pos_weight`.
   - The classifier-only sweep does set `scale_pos_weight=n_neg/n_pos`.

5. SVM probability semantics differ.
   - Factory SVM uses `probability=False`; original readout can be post-hoc calibrated.
   - Classifier-only SVM uses `probability=True`.
   - Any threshold rule must be selected from inner-CV OOF scores only, regardless of score type.

6. Current compact `logreg_l2` grid hit its lower boundary in all folds.
   - This does not invalidate the result.
   - It does mean the next search should expand lower and around `C=0.001`.

7. The threshold procedure is outer-test safe but still an internal CV operating-point analysis.
   - It is safe for internal reporting.
   - It is not an externally validated deployment threshold.

## Integration rule

Integrate classifier-only readouts downstream of saved fold latents and fold splits. Do not modify VAE training, tensor construction, metadata, or ledger logic.
"""
    (OUT_DIR / "code_risk_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    require(CLASSIFIERS_PY)
    require(SWEEP_SCRIPT)
    require(THRESHOLD_AUDIT_DIR / "command_log.json")
    require(SWEEP_DIR / "classifier_sweep_model_status.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(canonical_factory_rows() + classifier_only_rows())
    summary.to_csv(OUT_DIR / "classifier_factory_summary.csv", index=False)

    plan = pd.DataFrame(refined_plan_rows())
    plan.to_csv(OUT_DIR / "recommended_refined_sweep_plan.csv", index=False)

    context = load_context()
    write_readme(context)
    write_code_notes()

    command_log = {
        "script": str(Path(__file__).resolve()),
        "classifiers_py": str(CLASSIFIERS_PY),
        "classifier_only_sweep_script": str(SWEEP_SCRIPT),
        "threshold_audit_dir": str(THRESHOLD_AUDIT_DIR),
        "classifier_sweep_dir": str(SWEEP_DIR),
        "output_dir": str(OUT_DIR),
        "vae_retrained": False,
        "training_run": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(command_log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
