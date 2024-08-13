import pandas as pd
import numpy as np
import os

# Create the output directory if it doesn't exist
output_dir = 'Part13_outputs'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV files
XGB_df = pd.read_csv('Part11_XGB_output/Part11_XGB_input_top_30_features.csv', index_col='index_name')
XGB_RF_df = pd.read_csv('Part11_XGB_output_RF/Part11_XGB_input_top_30_features.csv', index_col='index_name')
RF_df = pd.read_csv('Part10_RF_output/Part10_RF_input_top_30_features.csv', index_col='index_name')
RF_RF_df = pd.read_csv('Part10_RF_output_RF/Part10_RF_input_top_30_features.csv', index_col='index_name')

# Get the feature names (excluding 'Case1_Control0')
XGB_features = [col for col in XGB_df.columns if col != 'Case1_Control0']
XGB_RF_features = [col for col in XGB_RF_df.columns if col != 'Case1_Control0']
RF_features = [col for col in RF_df.columns if col != 'Case1_Control0']
RF_RF_features = [col for col in RF_RF_df.columns if col != 'Case1_Control0']

# Find common features across all datasets
common_features = set(XGB_features) & set(XGB_RF_features) & set(RF_features) & set(RF_RF_features)

# Print results
print(f"Number of features in XGB dataset: {len(XGB_features)}")
print(f"Number of features in XGB_RF dataset: {len(XGB_RF_features)}")
print(f"Number of features in RF dataset: {len(RF_features)}")
print(f"Number of features in RF_RF dataset: {len(RF_RF_features)}")
print(f"Number of common features across all datasets: {len(common_features)}")
print("Common features:")
print(', '.join(sorted(common_features)))

# Save the common features to a text file
with open(os.path.join(output_dir, 'common_features.txt'), 'w') as f:
    f.write('\n'.join(sorted(common_features)))

# Combine datasets with all features
combined_all_df = pd.concat([XGB_df, XGB_RF_df, RF_df, RF_RF_df], axis=1)

# Remove duplicate 'Case1_Control0' column
combined_all_df = combined_all_df.loc[:, ~combined_all_df.columns.duplicated()]

# Move Case1_Control0 to the end
case_control = combined_all_df.pop('Case1_Control0')
combined_all_df['Case1_Control0'] = case_control

# Check for differences in values for common features
differences = []
for col in common_features:
    if not np.array_equal(XGB_df[col], XGB_RF_df[col]) or \
       not np.array_equal(XGB_df[col], RF_df[col]) or \
       not np.array_equal(XGB_df[col], RF_RF_df[col]):
        diff_rows = XGB_df.index[
            (XGB_df[col] != XGB_RF_df[col]) | 
            (XGB_df[col] != RF_df[col]) | 
            (XGB_df[col] != RF_RF_df[col])
        ]
        differences.append(f"Differences found in column '{col}' for rows: {list(diff_rows)}")

# Save differences to a text file
with open(os.path.join(output_dir, 'feature_differences.txt'), 'w') as f:
    f.write('\n'.join(differences))

combined_all_df.to_csv(os.path.join(output_dir, 'combined_all_features.csv'))
print(f"Combined dataset with all features saved as '{os.path.join(output_dir, 'combined_all_features.csv')}'")

# Create dataset with only common features
common_cols = list(common_features) + ['Case1_Control0']
combined_common_df = combined_all_df[common_cols]

combined_common_df.to_csv(os.path.join(output_dir, 'combined_common_features.csv'))
print(f"Combined dataset with common features saved as '{os.path.join(output_dir, 'combined_common_features.csv')}'")

# Save summary statistics to a text file
with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
    f.write(f"Number of features in XGB dataset: {len(XGB_features)}\n")
    f.write(f"Number of features in XGB_RF dataset: {len(XGB_RF_features)}\n")
    f.write(f"Number of features in RF dataset: {len(RF_features)}\n")
    f.write(f"Number of features in RF_RF dataset: {len(RF_RF_features)}\n")
    f.write(f"Number of common features across all datasets: {len(common_features)}\n")