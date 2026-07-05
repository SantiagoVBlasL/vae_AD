import re
from pathlib import Path
import pandas as pd

LOGS = sorted(Path("logs").glob("horizon4480_cycles56_full_5x5_*.log"))
if not LOGS:
    raise SystemExit("No encontre logs/horizon4480_cycles56_full_5x5_*.log")

log = LOGS[-1]
text = log.read_text(errors="ignore")

rows = []
pat = re.compile(r"RESULTADOS Fold (\d+)/5 \[(logreg|svm)\]: AUC=([0-9.]+), Acc=([0-9.]+)")
for fold, model, auc, acc in pat.findall(text):
    rows.append({
        "fold": int(fold),
        "stageA_model": model,
        "horizon4480_stageA_auc": float(auc),
        "stageA_acc": float(acc),
    })

cur = pd.DataFrame(rows)
if cur.empty:
    print("Todavia no hay resultados Stage A en el log.")
    print(f"Log: {log}")
    raise SystemExit(0)

cur_logreg = cur[cur["stageA_model"] == "logreg"].copy()

locked_path = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5_comparison/"
    "foldwise_comparison.csv"
)

if not locked_path.exists():
    raise SystemExit(f"No existe locked foldwise comparison: {locked_path}")

locked = pd.read_csv(locked_path)
locked = locked[
    (locked["run_id"] == "current_locked_ch1_0_2") &
    (locked["model_name"] == "logreg_l2") &
    (locked["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
][["fold", "auc", "pr_auc", "balanced_accuracy", "f1"]].rename(columns={
    "auc": "locked_stageB_auc",
    "pr_auc": "locked_stageB_pr_auc",
    "balanced_accuracy": "locked_stageB_BA",
    "f1": "locked_stageB_F1",
})

score = locked.merge(
    cur_logreg[["fold", "horizon4480_stageA_auc"]],
    on="fold",
    how="left"
)

score["delta_informal_auc"] = score["horizon4480_stageA_auc"] - score["locked_stageB_auc"]

def flag(x):
    if pd.isna(x):
        return "pending"
    if x > 0.015:
        return "WIN_strong"
    if x > 0.000:
        return "WIN_small"
    if x > -0.015:
        return "tie"
    return "LOSS"

score["battle_status_informal"] = score["delta_informal_auc"].apply(flag)

print("\n=== LIVE INFORMAL BATTLEBOARD ===")
print("Candidate: horizon4480_cycles56 FULL")
print("Important: comparing candidate Stage A logreg vs locked Stage B logreg_l2.")
print("Official decision requires candidate Stage B after all folds.\n")
print(score.to_string(index=False))

done = score.dropna(subset=["horizon4480_stageA_auc"]).copy()
if not done.empty:
    mean_candidate = done["horizon4480_stageA_auc"].mean()
    mean_locked_same_folds = done["locked_stageB_auc"].mean()
    print("\nCompleted folds only:")
    print(f"  candidate Stage A mean AUC: {mean_candidate:.6f}")
    print(f"  locked Stage B mean AUC same folds: {mean_locked_same_folds:.6f}")
    print(f"  informal delta: {mean_candidate - mean_locked_same_folds:+.6f}")

print("\nLocked global targets:")
print("  AUC target    : 0.778785")
print("  PR-AUC target : 0.551832")
print("\nLog:", log)
