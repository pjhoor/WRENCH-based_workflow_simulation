$files = @(
  "data\merged_data\merged_data_low_tier.csv",
  "data\merged_data\merged_data_mid_tier.csv",
  "data\merged_data\merged_data_high_tier.csv",
  "data\merged_data\merged_data_extra_high_tier.csv"
)
$ErrorActionPreference = "Stop"

foreach ($file in $files) {
  python myPython\modelTraining.py $file
}