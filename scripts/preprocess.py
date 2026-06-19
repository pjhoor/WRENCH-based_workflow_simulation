import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _featureExtractor import main as run_feature_extractor
from _executionProcessor import create_workflow_summary
from _dataMerger import run_merger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full feature-extraction → execution-processing → merge pipeline."
    )
    parser.add_argument(
        "--execution_output",
        type=Path,
        required=True,
        help="Path to the execution output CSV (input for executionProcessor.py).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    execution_output_file   = args.execution_output
    data_dir = execution_output_file.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_features_file = data_dir / "extracted_features.csv"
    execution_per_workflow  = data_dir / "execution_per_workflow.csv"

    suffix = execution_output_file.stem.removeprefix("execution_output_")
    merged_data_file = data_dir / f"merged_data/merged_data_{suffix}.csv"

    print("=" * 60)
    print("STEP 1 – Feature Extraction")
    print("=" * 60)
    run_feature_extractor()

    print("\n" + "=" * 60)
    print("STEP 2 – Execution Processing")
    print(f"  Input : {execution_output_file}")
    print(f"  Output: {execution_per_workflow}")
    print("=" * 60)
    create_workflow_summary(execution_output_file, execution_per_workflow)

    print("\n" + "=" * 60)
    print("STEP 3 – Data Merge")
    print("=" * 60)
    run_merger(extracted_features_file, execution_per_workflow, merged_data_file)

    print("\nDone.")
    print(f"    Final output: {merged_data_file.resolve()}")

if __name__ == "__main__":
    main()
