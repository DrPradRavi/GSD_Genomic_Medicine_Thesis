import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'FS_Merge_Output'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV files
turf_df = pd.read_csv('FS_TuRF/FS_TuRF_selected_features_dataset.csv', index_col=0)
rf_df = pd.read_csv('FS_RF/FS_RF_selected_features_dataset.csv', index_col=0)

# Count features in each dataset
turf_features = set(turf_df.columns)
rf_features = set(rf_df.columns)

# Calculate unique and common features
unique_turf = turf_features - rf_features
unique_rf = rf_features - turf_features
common_features = turf_features.intersection(rf_features)

# Merge the dataframes
merged_df = pd.concat([turf_df, rf_df], axis=1)

# Remove duplicate columns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Sort columns alphabetically
merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(os.path.join(output_dir, 'merged_features.csv'))

# Prepare output information
output_info = [
    f"Total features in TuRF: {len(turf_features)}",
    f"Total features in RF: {len(rf_features)}",
    f"Unique features in TuRF: {len(unique_turf)}",
    f"Unique features in RF: {len(unique_rf)}",
    f"Common features: {len(common_features)}",
    f"Total unique features: {len(merged_df.columns)}",
    "Merged CSV file created: merged_features.csv"
]

# Save output information to a text file
with open(os.path.join(output_dir, 'merge_summary.txt'), 'w') as f:
    for line in output_info:
        f.write(line + '\n')

# Print output information to console
for line in output_info:
    print(line)