#!/usr/bin/env bash

platform="platforms/low_tier.xml"
workflow_dir="workflows"

echo "Running all .json files (recipes) recursively in the '$workflow_dir' folder:"

find "$workflow_dir" -type f -name "*.json" -print0 | sort -z | while IFS= read -r -d $'\0' recipe_path; do
    recipe_file=$(basename "$recipe_path")
    recipe_dir=$(dirname "$recipe_path")

    echo "  - Running recipe: '$recipe_file' in the folder: '$recipe_dir'"
    ./build/my-wrench-simulator --wrench-commport-pool-size=200000 "$platform" "$recipe_path" --wrench-energy-simulation

    if [ $? -ne 0 ]; then
        echo "    Error running recipe '$recipe_file' in folder: '$recipe_dir'."
    fi
done

echo ""
echo "------------------------------------------------------------------"
echo "End of execution of recipes found recursively in '$workflow_dir'.."

exit 0