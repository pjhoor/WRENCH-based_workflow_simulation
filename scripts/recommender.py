import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _featureExtractor import extract_features
from modelTraining import load_model, FEATURE_COLUMNS

CLUSTERS = [
    "low_tier",
    "mid_tier",
    "high_tier",
    "extra_high_tier",
]

MODEL_FILE_MAP = {
    "lr":       {"ct": "lr_compute_time.pkl",       "tp": "lr_total_power.pkl"},
    "pr":       {"ct": "pr_compute_time.pkl",       "tp": "pr_total_power.pkl"},
    "er":       {"ct": "er_compute_time.pkl",       "tp": "er_total_power.pkl"},
    "poly":     {"ct": "poly_compute_time.pkl",     "tp": "poly_total_power.pkl",
                 "ct_feat": "poly_features_compute_time.pkl",
                 "tp_feat": "poly_features_total_power.pkl"},
    "rf":       {"ct": "rf_compute_time.pkl",       "tp": "rf_total_power.pkl"},
    "xgb":      {"ct": "xgb_compute_time.pkl",      "tp": "xgb_total_power.pkl"},
    "gbr":      {"ct": "gbr_compute_time.pkl",      "tp": "gbr_total_power.pkl"},
    "catboost": {"ct": "catboost_compute_time.pkl", "tp": "catboost_total_power.pkl"},
    "lgbm":     {"ct": "lgbm_compute_time.pkl",     "tp": "lgbm_total_power.pkl"},
    "sr":       {"ct": "sr_compute_time.pkl",       "tp": "sr_total_power.pkl"},
}

MODEL_LABELS = {
    "lr":       "Linear Regression",
    "pr":       "Power Regression",
    "er":       "Exponential Regression",
    "poly":     "Polynomial Regression",
    "rf":       "Random Forest",
    "xgb":      "XGBoost",
    "gbr":      "Gradient Boosting",
    "catboost": "CatBoost",
    "lgbm":     "LightGBM",
    "sr":       "Symbolic Regression",
}

def list_workflows(workflows_dir: str) -> list[str]:
    workflows_path = Path(workflows_dir)
    if not workflows_path.exists():
        print(f"Error: workflows directory not found: {workflows_dir}")
        sys.exit(1)

    all_json_files = sorted(workflows_path.rglob("*.json"))

    if not all_json_files:
        print(f"Error: no .json workflow files found under '{workflows_dir}'.")
        sys.exit(1)

    return [str(p) for p in all_json_files]

def _extract_workflow_number(path: Path) -> int:
    match = re.search(r"-workflow-(\d+)-0\.json$", path.name)
    return int(match.group(1)) if match else 0

def list_workflows_bulk(workflows_dir: str) -> list[str]:
    workflows_path = Path(workflows_dir)
    if not workflows_path.exists():
        print(f"Error: workflows directory not found: {workflows_dir}")
        sys.exit(1)

    type_dirs = sorted([d for d in workflows_path.iterdir() if d.is_dir()])

    if not type_dirs:
        print(f"Error: no workflow-type subdirectories found under '{workflows_dir}'.")
        sys.exit(1)

    selected: list[str] = []

    for type_dir in type_dirs:
        json_files = sorted(
            type_dir.glob("*.json"),
            key=_extract_workflow_number,
        )

        if not json_files:
            print(f"  WARNING: no .json files found in '{type_dir}', skipping.")
            continue

        every_other = json_files[::2][:100]
        selected.extend(str(p) for p in every_other)

        print(
            f"  {type_dir.name:<30} {len(every_other):>3} workflows selected "
            f"(first: {every_other[0].name}, last: {every_other[-1].name})"
        )

    if not selected:
        print(f"Error: no workflow files could be selected under '{workflows_dir}'.")
        sys.exit(1)

    return selected

def pick_random_workflow(workflows_dir: str) -> str:
    all_json_files = list_workflows(workflows_dir)
    chosen = random.choice(all_json_files)
    print(f"Randomly selected workflow: {chosen}")
    return chosen

def get_feature_row(workflow_path: str, label_encoder, models_dir: str) -> pd.DataFrame:
    wf_file = Path(workflow_path)
    with open(wf_file, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    # Pass Path object so extract_features can read wf_file.parent.name
    features = extract_features(workflow_json, wf_file)

    print("Extracted features:")
    for col in FEATURE_COLUMNS:
        print(f"  {col:22s}: {features.get(col, 'N/A')}")

    row = {col: features[col] for col in FEATURE_COLUMNS}

    # Encode workflow_type string → integer using the saved LabelEncoder
    row["workflow_type"] = label_encoder.transform([row["workflow_type"]])[0]

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

def predict_for_cluster(
    X: pd.DataFrame,
    model_key: str,
    cluster: str,
    models_dir: str,
) -> tuple[float, float]:
    cluster_dir = os.path.join(models_dir, cluster)
    files = MODEL_FILE_MAP[model_key]

    ct_path = os.path.join(cluster_dir, files["ct"])
    tp_path = os.path.join(cluster_dir, files["tp"])

    for path in (ct_path, tp_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Make sure models for the '{cluster}' cluster have been trained "
                f"with model_training.py."
            )

    ct_model = load_model(files["ct"], models_dir=cluster_dir)
    tp_model = load_model(files["tp"], models_dir=cluster_dir)

    if model_key == "poly":
        ct_poly = load_model(files["ct_feat"], models_dir=cluster_dir)
        tp_poly = load_model(files["tp_feat"], models_dir=cluster_dir)
        X_ct = ct_poly.transform(X)
        X_tp = tp_poly.transform(X)
    else:
        X_ct = X
        X_tp = X

    compute_time = float(ct_model.predict(X_ct)[0])
    total_power  = float(tp_model.predict(X_tp)[0])

    compute_time = max(compute_time, 0.0)
    total_power  = max(total_power,  0.0)

    return compute_time, total_power

def recommend(
    workflow_path: str,
    model_key: str,
    models_dir: str,
    *,
    verbose: bool = True,
) -> list[tuple[str, dict]]:
    if verbose:
        print("=" * 60)
        print("WORKFLOW ENERGY RECOMMENDER")
        print("=" * 60)
        print(f"  Workflow  : {workflow_path}")
        print(f"  Model     : {MODEL_LABELS[model_key]} ({model_key})")
        print(f"  Models dir: {models_dir}")
        print()

    # Load the label encoder from the first available cluster's models dir,
    # falling back to the root models_dir
    encoder_path = None
    for cluster in CLUSTERS:
        candidate = os.path.join(models_dir, cluster, "workflow_type_label_encoder.pkl")
        if os.path.exists(candidate):
            encoder_path = candidate
            break
    if encoder_path is None:
        candidate = os.path.join(models_dir, "workflow_type_label_encoder.pkl")
        if os.path.exists(candidate):
            encoder_path = candidate

    if encoder_path is None:
        print("Error: workflow_type_label_encoder.pkl not found in models directory.")
        sys.exit(1)

    label_encoder = load_model(os.path.basename(encoder_path),
                               models_dir=os.path.dirname(encoder_path))

    X = get_feature_row(workflow_path, label_encoder, models_dir)

    if verbose:
        print("\nPredictions per cluster:")
        print(f"  {'Cluster':<24} {'Compute Time':>14} {'Total Power':>13} {'Energy (CT×TP)':>16}")
        print("  " + "-" * 70)

    results: dict[str, dict] = {}
    for cluster in CLUSTERS:
        try:
            ct, tp = predict_for_cluster(X, model_key, cluster, models_dir)
            energy = ct * tp
            results[cluster] = {"compute_time": ct, "total_power": tp, "energy": energy}
            if verbose:
                print(f"  {cluster:<24} {ct:>14.4f} {tp:>13.4f} {energy:>16.4f}")
        except FileNotFoundError as e:
            if verbose:
                print(f"  {cluster:<24}  [SKIPPED - {e}]")

    if not results:
        print("\nNo cluster models could be loaded. Aborting.")
        sys.exit(1)

    ranked = sorted(results.items(), key=lambda x: x[1]["energy"])

    # Attach all per-cluster predictions to the best entry for downstream use
    ranked[0][1]["all_results"] = results

    if verbose:
        best_cluster, best = ranked[0]
        print()
        print("=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print(f"     Most energy-efficient cluster : {best_cluster.upper()}")
        print(f"     Predicted compute time         : {best['compute_time']:.4f}")
        print(f"     Predicted total power          : {best['total_power']:.4f}")
        print(f"     Predicted energy consumption   : {best['energy']:.4f}")
        print()
        print("  Ranked by energy consumption (lowest → highest):")
        for rank, (cluster, vals) in enumerate(ranked, start=1):
            marker = " ← best" if rank == 1 else ""
            print(f"    {rank}. {cluster:<24}  energy = {vals['energy']:.4f}{marker}")
        print("=" * 60)

    return ranked

def run_batch(
    workflows_dir: str,
    model_key: str,
    models_dir: str,
    output_csv: str = "batch_choices.csv",
) -> None:
    print("=" * 60)
    print("BATCH MODE")
    print("=" * 60)
    print(f"  Workflows dir  : {workflows_dir}")
    print(f"  Model          : {MODEL_LABELS[model_key]} ({model_key})")
    print(f"  Models dir     : {models_dir}")
    print(f"  Selection      : every other workflow per type (100 × 7 = 700)")
    print(f"  Output CSV     : {output_csv}")
    print("=" * 60)
    print()

    workflow_paths = list_workflows_bulk(workflows_dir)
    total = len(workflow_paths)

    print()
    print(f"  Total workflows selected: {total}")
    print()

    rank_counts: dict[str, dict[int, int]] = {
        cluster: defaultdict(int) for cluster in CLUSTERS
    }
    failed = 0

    dataset_rows: list[dict] = []

    for idx, wf_path in enumerate(workflow_paths, start=1):
        print(f"[{idx}/{total}] {wf_path}")
        try:
            ranked = recommend(wf_path, model_key, models_dir, verbose=False)
            for pos, (cluster, _) in enumerate(ranked, start=1):
                rank_counts[cluster][pos] += 1

            best_cluster, best_metrics = ranked[0]
            all_results = best_metrics.get("all_results", {})

            row: dict = {
                "workflow_path":             wf_path,
                "model":                     model_key,
                "recommended_cluster":       best_cluster,
                "recommended_compute_time":  best_metrics["compute_time"],
                "recommended_total_power":   best_metrics["total_power"],
                "recommended_energy":        best_metrics["energy"],
            }
            for cluster in CLUSTERS:
                if cluster in all_results:
                    row[f"{cluster}_compute_time"] = all_results[cluster]["compute_time"]
                    row[f"{cluster}_total_power"]  = all_results[cluster]["total_power"]
                    row[f"{cluster}_energy"]       = all_results[cluster]["energy"]
                else:
                    row[f"{cluster}_compute_time"] = None
                    row[f"{cluster}_total_power"]  = None
                    row[f"{cluster}_energy"]       = None

            dataset_rows.append(row)

        except Exception as e:
            print(f"  WARNING: skipped due to error - {e}")
            failed += 1

    if dataset_rows:
        df = pd.DataFrame(dataset_rows)
        df.to_csv(output_csv, index=False)
        print(f"\n  Dataset saved → {output_csv}  ({len(df)} rows)")
    else:
        print("\n  No successful results to save.")

    processed = total - failed
    print()
    print("=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"  Workflows processed : {processed} / {total}")
    if failed:
        print(f"  Workflows failed    : {failed}")
    print()

    col_w = 24
    print(f"  {'Cluster':<{col_w}} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6}  {'% 1st':>7}")
    print("  " + "-" * (col_w + 36))

    for cluster in CLUSTERS:
        counts = rank_counts[cluster]
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        c3 = counts.get(3, 0)
        c4 = counts.get(4, 0)
        pct = (c1 / processed * 100) if processed > 0 else 0.0
        print(f"  {cluster:<{col_w}} {c1:>6} {c2:>6} {c3:>6} {c4:>6}  {pct:>6.1f}%")

    print()

    best_overall = max(CLUSTERS, key=lambda c: rank_counts[c].get(1, 0))
    print(f"     Most frequently recommended: {best_overall.upper()}"
          f"  ({rank_counts[best_overall].get(1, 0)}× first place)")
    print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend the most energy-efficient cluster for a workflow."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--workflow",
        type=str,
        metavar="WORKFLOW",
        help="Path to a single workflow JSON file.",
    )
    source_group.add_argument(
        "--random",
        action="store_true",
        help="Pick a random workflow JSON from --workflows_dir.",
    )
    source_group.add_argument(
        "--batch",
        action="store_true",
        help="Process 100 workflows per type (every other, 700 total) under "
             "--workflows_dir and report ranking statistics.",
    )

    parser.add_argument(
        "--workflows_dir",
        type=str,
        default="workflows",
        help="Root directory to search for workflow JSON files when using --random "
             "or --batch (default: workflows/). Expected to contain one subfolder per "
             "workflow type, each with 200 workflow JSON files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_FILE_MAP.keys()),
        default="rf",
        help="ML model to use for prediction (default: rf). "
             "Choices: lr, pr, er, poly, rf, xgb, gbr, catboost, lgbm, sr.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Root directory containing per-cluster model subdirectories "
             "(default: models/). Expected layout: "
             "models/low_tier_cluster/, models/med_tier_cluster/, etc.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="batch_choices.csv",
        help="Path to save the batch results CSV (default: batch_choices.csv). "
             "Only used with --batch.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.batch:
        run_batch(args.workflows_dir, args.model, args.models_dir, args.output_csv)
    elif args.random:
        workflow_path = pick_random_workflow(args.workflows_dir)
        recommend(workflow_path, args.model, args.models_dir)
    else:
        recommend(args.workflow, args.model, args.models_dir)
