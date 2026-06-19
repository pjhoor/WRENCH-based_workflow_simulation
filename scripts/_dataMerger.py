import pandas as pd
from pathlib import Path

features_from_file1 = [
    'workflow_filename',
    'workflow_type',
    'avg_file_size',
    'task_count',
    'edge_count',
    'unique_task_types',
    'workflow_depth'
]

def run_merger(input_file_1, input_file_2, output_file):
    root_dir = Path(__file__).parent.parent

    df1 = pd.read_csv(input_file_1)
    df2 = pd.read_csv(input_file_2)

    df1_selected = df1[features_from_file1]

    merged_df = pd.merge(df1_selected, df2, on='workflow_filename', how='inner')

    merged_df.to_csv(output_file, index=False)

    print(f"Merge complete! Output saved to: {output_file}")
    print(f"Total rows merged: {len(merged_df)}")