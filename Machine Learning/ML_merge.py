import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'ML_Merge_Output'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV files
rf_df = pd.read_csv('ML_RF/ML_RF_selected_features_dataset.csv', index_col=0)
xgb_df = pd.read_csv('ML_XGB/ML_XGB_selected_features_dataset.csv', index_col=0)

# Count features in each dataset
rf_features = set(rf_df.columns)
xgb_features = set(xgb_df.columns)

# Calculate unique and common features
unique_rf = rf_features - xgb_features
unique_xgb = xgb_features - rf_features
common_features = rf_features.intersection(xgb_features)

# Merge the dataframes
merged_df = pd.concat([rf_df, xgb_df], axis=1)

# Remove duplicate columns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Sort columns alphabetically
merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(os.path.join(output_dir, 'merged_features.csv'))

# Prepare output information
output_info = [
    f"Total features in RF: {len(rf_features)}",
    f"Total features in XGB: {len(xgb_features)}",
    f"Unique features in RF: {len(unique_rf)}",
    f"Unique features in XGB: {len(unique_xgb)}",
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
