import pandas as pd
from pathlib import Path

def create_workflow_summary(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    summary_data = []
    
    for workflow, group in df.groupby('workflow_filename'):
        compute_time = group['compute_time'].iloc[0]
        
        total_power = group['power'].sum()
        
        summary_data.append({
            'workflow_filename': workflow,
            'compute_time': compute_time,
            'total_power': total_power
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_df.to_csv(output_csv, index=False)
    
    print(f"Summary created successfully!")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    print(f"\nSummary Statistics:")
    print(f"Number of unique workflows: {len(summary_df)}")
    print(f"\n{summary_df}")

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    input_file = root_dir / "data/execution_output.csv"
    output_file = root_dir / "data/execution_per_workflow.csv"
    
    create_workflow_summary(input_file, output_file)