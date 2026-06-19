import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict, deque

def as_id_list(value):
    if not value:
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            if "id" in item:
                out.append(item["id"])
    return out

def get_task_id(task, fallback_index):
    return task.get("id") or task.get("name") or f"task_{fallback_index}"

def get_task_type(task):
    return task.get("name") or task.get("type") or task.get("id") or "unknown"

def get_optional_cores(task):
    for key in ("coreCount", "cores", "numCores", "num_cores"):
        if key in task and task[key] is not None:
            return task[key]
    return None

def extract_features(workflow_json, workflow_file):
    spec = workflow_json.get("workflow", {}).get("specification", {})
    tasks = spec.get("tasks", []) or []
    files = spec.get("files", []) or []

    task_ids = []
    task_by_id = {}
    parents_map = {}
    children_map = {}
    task_types = defaultdict(int)
    declared_cores = []

    for i, task in enumerate(tasks):
        tid = get_task_id(task, i)
        task_ids.append(tid)
        task_by_id[tid] = task
        task_types[get_task_type(task)] += 1

        parents = as_id_list(task.get("parents", []))
        children = as_id_list(task.get("children", []))
        parents_map[tid] = parents
        children_map[tid] = children

        cores = get_optional_cores(task)
        if cores is not None:
            declared_cores.append(cores)

    file_sizes = []
    zero_size_files = 0
    for f in files:
        size = f.get("sizeInBytes", 0) if isinstance(f, dict) else 0
        file_sizes.append(size)
        if size == 0:
            zero_size_files += 1

    input_file_refs = 0
    output_file_refs = 0
    total_input_bytes = 0
    total_output_bytes = 0

    file_size_by_id = {}
    for i, f in enumerate(files):
        if isinstance(f, dict):
            fid = f.get("id", f"file_{i}")
            file_size_by_id[fid] = f.get("sizeInBytes", 0)

    for task in tasks:
        for fid in as_id_list(task.get("inputFiles", [])):
            input_file_refs += 1
            total_input_bytes += file_size_by_id.get(fid, 0)
        for fid in as_id_list(task.get("outputFiles", [])):
            output_file_refs += 1
            total_output_bytes += file_size_by_id.get(fid, 0)

    edge_count = sum(len(v) for v in children_map.values())
    in_degrees = [len(parents_map[tid]) for tid in task_ids]
    out_degrees = [len(children_map[tid]) for tid in task_ids]

    indeg = {tid: len(parents_map[tid]) for tid in task_ids}
    dist = {tid: 1 for tid in task_ids}
    q = deque([tid for tid in task_ids if indeg[tid] == 0])

    while q:
        u = q.popleft()
        for v in children_map.get(u, []):
            if v not in dist:
                dist[v] = 1
            dist[v] = max(dist[v], dist[u] + 1)
            if v in indeg:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    workflow_depth = max(dist.values()) if dist else 0

    features = {
        "workflow_type": workflow_file.parent.name,
        "task_count": len(tasks),
        "file_count": len(files),
        "unique_task_types": len(task_types),
        "edge_count": edge_count,
        "workflow_depth": workflow_depth,
        "avg_in_degree": round(statistics.mean(in_degrees), 3) if in_degrees else 0,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "avg_out_degree": round(statistics.mean(out_degrees), 3) if out_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "total_input_file_refs": input_file_refs,
        "total_output_file_refs": output_file_refs,
        "total_input_bytes": total_input_bytes,
        "total_output_bytes": total_output_bytes,
        "avg_file_size": round(statistics.mean(file_sizes), 3) if file_sizes else 0,
        "max_file_size": max(file_sizes) if file_sizes else 0,
        "min_file_size": min(file_sizes) if file_sizes else 0,
        "zero_size_file_count": zero_size_files,
        "declared_core_count_avg": round(statistics.mean(declared_cores), 3) if declared_cores else 0,
        "declared_core_count_max": max(declared_cores) if declared_cores else 0,
    }

    for ttype, count in task_types.items():
        features[f"type_count__{ttype}"] = count

    return features

def find_workflow_files(root_path):
    return sorted(Path(root_path).rglob("*.json"))

def main():
    root_dir = Path(__file__).parent.parent
    workflows_dir = root_dir / "workflows"
    output_file = root_dir / "data" / "extracted_features.csv"

    if not workflows_dir.exists():
        print(f"Workflows directory not found: {workflows_dir}")
        return

    workflow_files = find_workflow_files(workflows_dir)
    print(f"Found {len(workflow_files)} workflow files")

    all_columns = {"workflow_filename"}
    for wf_file in workflow_files:
        try:
            with open(wf_file, "r", encoding="utf-8") as f:
                workflow_json = json.load(f)
            features = extract_features(workflow_json, wf_file)
            all_columns.update(features.keys())
        except Exception as e:
            print(f"✗ Failed to inspect: {wf_file} ({e})")

    fieldnames = ["workflow_filename"] + sorted(c for c in all_columns if c != "workflow_filename")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    error_count = 0
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for wf_file in workflow_files:
            try:
                with open(wf_file, "r", encoding="utf-8") as f:
                    workflow_json = json.load(f)

                features = extract_features(workflow_json, wf_file)
                row = {"workflow_filename": wf_file.name}
                row.update(features)
                writer.writerow(row)
                print(f"Processed: {row['workflow_filename']}")
            except Exception as e:
                error_count += 1
                print(f"Failed to extract features from: {wf_file} ({e})")

    print(f"\nSuccesfully processed {len(workflow_files) - error_count}/{len(workflow_files)} workflows.")
    print(f"\nFeatures saved to: {output_file}")

if __name__ == "__main__":
    main()
