#!/usr/bin/env bash

platforms=(
  "platforms/low_tier.xml"
  "platforms/mid_tier.xml"
  "platforms/high_tier.xml"
  "platforms/extra_high_tier.xml"
)

workflow_dir="workflows"

overall_failures=0
overall_runs=0

mapfile -d '' -t recipes < <(find "$workflow_dir" -type f -name "*.json" -print0 | sort -z)

if [ ${#recipes[@]} -eq 0 ]; then
    echo "No .json recipes found in '$workflow_dir'."
    exit 0
fi

for platform in "${platforms[@]}"; do
    echo ""
    echo "=================================================================="
    echo "Running recipes for platform: $platform"
    echo "=================================================================="

    for recipe_path in "${recipes[@]}"; do
        recipe_file=$(basename "$recipe_path")
        recipe_dir=$(dirname "$recipe_path")

        echo "  - Running recipe: '$recipe_file' in folder: '$recipe_dir'"
        # ./build/my-wrench-simulator --wrench-commport-pool-size=20000 "$platform" "$recipe_path" --wrench-energy-simulation
        ./build/my-wrench-simulator --wrench-commport-pool-size=100000 "$platform" "$recipe_path" --wrench-energy-simulation
        rc=$?
        overall_runs=$((overall_runs+1))
        if [ $rc -ne 0 ]; then
        echo "    Error (exit $rc) running recipe '$recipe_file' for platform '$platform'."
        overall_failures=$((overall_failures+1))
        fi
    done
done

echo ""
echo "------------------------------------------------------------------"
echo "Summary: ran $overall_runs recipes across ${#platforms[@]} platforms. Failures: $overall_failures"

if [ $overall_failures -ne 0 ]; then
    exit 1
fi

exit 0