import pathlib
from wfcommons.wfchef.recipes import (
    BlastRecipe,
    BwaRecipe,
    CyclesRecipe,
    EpigenomicsRecipe,
    GenomeRecipe,
    MontageRecipe,
    SeismologyRecipe,
)
from wfcommons import WorkflowGenerator

WORKFLOWS = [
    (BlastRecipe,       "blast",       "blast",       60),
    (BwaRecipe,         "bwa",         "bwa",         106),
    (CyclesRecipe,      "cycles",      "cycles",      69),
    (EpigenomicsRecipe, "epigenomics", "epigenomics", 60),
    (GenomeRecipe,      "genome",      "genome",      54),
    (MontageRecipe,     "montage",     "montage",     60),
    (SeismologyRecipe,  "seismology",  "seismology",  103),
]

print("=== Workflow Generator ===\n")
amounts = {}
for _, _, prefix, _ in WORKFLOWS:
    amounts[prefix] = 200
    
base_dir = pathlib.Path(__file__).parent.parent / "workflows"

for recipe_cls, subfolder, prefix, min_tasks in WORKFLOWS:
    amount = amounts[prefix]
    if amount == 0:
        print(f"[{prefix}] Skipped (0 requested).")
        continue

    output_dir = base_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    print(f"[{prefix}] Generating {amount} workflow(s) starting at {min_tasks} tasks...")

    for amount_tasks in range(min_tasks, min_tasks + amount * 10, 10):
        generator = WorkflowGenerator(recipe_cls.from_num_tasks(amount_tasks))
        workflows = generator.build_workflows(1)

        for index, workflow in enumerate(workflows):
            output_path = output_dir / f"{prefix}-workflow-{amount_tasks}-{index}.json"
            try:
                workflow.write_json(output_path)
                count += 1
                print(f"  [{prefix}] Created {count}/{amount}  →  {output_path.name}")
            except Exception as e:
                print(f"  [{prefix}] Error writing {output_path.name}: {e}")

    print(f"[{prefix}] Done! ({count} workflow(s) written)\n")

print("=== All workflows generated! ===")
