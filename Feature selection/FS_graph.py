import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import cohen_kappa_score

# Constants
OUTPUT_DIR = "FS_Compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
rf_importance_df = pd.read_csv("FS_RF/FS_RF_all_features_importance.csv")
turf_importance_df = pd.read_csv("FS_TuRF/FS_TuRF_all_features_importance.csv")

# Merge data on feature
merged_df = pd.merge(rf_importance_df, turf_importance_df, on='feature', suffixes=('_RF', '_TuRF'))

# Load top features
rf_top_features = pd.read_csv("FS_RF/FS_RF_top_features.csv")
turf_top_features = pd.read_csv("FS_TuRF/FS_TuRF_selected_features.csv")

# Create category columns
merged_df['category'] = "Wasn't selected"
merged_df.loc[merged_df['feature'].isin(rf_top_features['feature']), 'category'] = 'RF selected'
merged_df.loc[merged_df['feature'].isin(turf_top_features['feature']), 'category'] = 'TuRF selected'
merged_df.loc[merged_df['feature'].isin(rf_top_features['feature']) & 
              merged_df['feature'].isin(turf_top_features['feature']), 'category'] = 'RF and TuRF selected'

# Calculate shared and unique features
rf_features_set = set(rf_top_features['feature'])
turf_features_set = set(turf_top_features['feature'])

shared_features = rf_features_set & turf_features_set
rf_unique_features = rf_features_set - turf_features_set
turf_unique_features = turf_features_set - rf_features_set

# Save comparison results to CSV
comparison_summary = {
    'Category': ['Shared', 'RF unique', 'TuRF unique'],
    'Count': [len(shared_features), len(rf_unique_features), len(turf_unique_features)]
}
comparison_df = pd.DataFrame(comparison_summary)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'Part12_RFvsTuRF_comparison_summary.csv'), index=False)

# Calculate Cohen's Kappa
rf_selected = merged_df['feature'].isin(rf_top_features['feature']).astype(int)
turf_selected = merged_df['feature'].isin(turf_top_features['feature']).astype(int)

kappa_score = cohen_kappa_score(rf_selected, turf_selected)
print(f"Cohen's Kappa Score: {kappa_score}")

# Save Cohen's Kappa result to CSV
kappa_df = pd.DataFrame({'Metric': ['Cohen\'s Kappa'], 'Score': [kappa_score]})
kappa_df.to_csv(os.path.join(OUTPUT_DIR, 'Part12_RFvsTuRF_kappa_score.csv'), index=False)

# Set up the plot
plt.figure(figsize=(12, 10))
plt.grid(True)

# Define categories, colors, markers, sizes, and alphas
categories = ["Wasn't selected", 'RF selected', 'TuRF selected', 'RF and TuRF selected']
colors = ['grey', 'red', 'blue', 'yellow']
markers = ['o', '+', 'x', '*']
sizes = [40, 50, 50, 65]
alphas = [0.4, 0.6, 0.6, 0.8]

# Create the scatter plot
for cat, color, marker, size, alpha in zip(categories, colors, markers, sizes, alphas):
    subset = merged_df[merged_df['category'] == cat]
    if cat in ['RF selected', 'TuRF selected']:
        plt.scatter(subset['importance_RF'], subset['importance_TuRF'], 
                    c=color, marker=marker, s=size, 
                    alpha=alpha, label=cat, linewidth=1)
    elif cat == 'RF and TuRF selected':
        plt.scatter(subset['importance_RF'], subset['importance_TuRF'], 
                    facecolors=color, edgecolors='black', marker=marker, s=size, 
                    alpha=alpha, label=cat, linewidth=0.3)
    else:
        plt.scatter(subset['importance_RF'], subset['importance_TuRF'], 
                    c=color, marker=marker, s=size, alpha=alpha, label=cat)

# Customize the plot
plt.xlabel('RF Importance Score')
plt.ylabel('TuRF Importance Score')
plt.title('Feature Importance Comparison: RF vs TuRF')
plt.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save and show the plot
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Part12_RFvsTuRF.png'))
plt.show()