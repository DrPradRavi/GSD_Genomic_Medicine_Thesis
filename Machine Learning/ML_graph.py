import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import cohen_kappa_score

# Constants
OUTPUT_DIR = "ML_Compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
rf_importance_df = pd.read_csv("ML_RF/ML_RF_all_features_importance.csv")
xgb_importance_df = pd.read_csv("ML_XGB/ML_XGB_all_features_importance.csv")

# Merge data on feature
merged_df = pd.merge(rf_importance_df, xgb_importance_df, on='feature', suffixes=('_RF', '_XGB'))

# Load top features
rf_top_features = pd.read_csv("ML_RF/ML_RF_top_features.csv")
xgb_top_features = pd.read_csv("ML_XGB/ML_XGB_top_features.csv")

# Create category columns
merged_df['category'] = "Wasn't selected"
merged_df.loc[merged_df['feature'].isin(rf_top_features['feature']), 'category'] = 'RF selected'
merged_df.loc[merged_df['feature'].isin(xgb_top_features['feature']), 'category'] = 'XGB selected'
merged_df.loc[merged_df['feature'].isin(rf_top_features['feature']) & 
              merged_df['feature'].isin(xgb_top_features['feature']), 'category'] = 'RF and XGB selected'

# Calculate shared and unique features
rf_features_set = set(rf_top_features['feature'])
xgb_features_set = set(xgb_top_features['feature'])

shared_features = rf_features_set & xgb_features_set
rf_unique_features = rf_features_set - xgb_features_set
xgb_unique_features = xgb_features_set - rf_features_set

# Save comparison results to CSV
comparison_summary = {
    'Category': ['Shared', 'RF unique', 'XGB unique'],
    'Count': [len(shared_features), len(rf_unique_features), len(xgb_unique_features)]
}
comparison_df = pd.DataFrame(comparison_summary)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'ML_RFvsXGB_comparison_summary.csv'), index=False)

# Calculate Cohen's Kappa
rf_selected = merged_df['feature'].isin(rf_top_features['feature']).astype(int)
xgb_selected = merged_df['feature'].isin(xgb_top_features['feature']).astype(int)

kappa_score = cohen_kappa_score(rf_selected, xgb_selected)
print(f"Cohen's Kappa Score: {kappa_score}")

# Save Cohen's Kappa result to CSV
kappa_df = pd.DataFrame({'Metric': ['Cohen\'s Kappa'], 'Score': [kappa_score]})
kappa_df.to_csv(os.path.join(OUTPUT_DIR, 'ML_RFvsXGB_kappa_score.csv'), index=False)

# Set up the plot
plt.figure(figsize=(12, 10))
plt.grid(True)

# Define categories, colors, markers, sizes, and alphas
categories = ["Wasn't selected", 'RF selected', 'XGB selected', 'RF and XGB selected']
colors = ['grey', 'red', 'blue', 'yellow']
markers = ['o', '+', 'x', '*']
sizes = [40, 50, 50, 65]
alphas = [0.4, 0.6, 0.6, 0.8]

# Create the scatter plot
for cat, color, marker, size, alpha in zip(categories, colors, markers, sizes, alphas):
    subset = merged_df[merged_df['category'] == cat]
    if cat in ['RF selected', 'XGB selected']:
        plt.scatter(subset['importance_RF'], subset['importance_XGB'], 
                    c=color, marker=marker, s=size, 
                    alpha=alpha, label=cat, linewidth=1)
    elif cat == 'RF and XGB selected':
        plt.scatter(subset['importance_RF'], subset['importance_XGB'], 
                    facecolors=color, edgecolors='black', marker=marker, s=size, 
                    alpha=alpha, label=cat, linewidth=0.3)
    else:
        plt.scatter(subset['importance_RF'], subset['importance_XGB'], 
                    c=color, marker=marker, s=size, alpha=alpha, label=cat)

# Customize the plot
plt.xlabel('RF Importance Score')
plt.ylabel('XGB Importance Score')
plt.title('Feature Importance Comparison: RF vs XGB')
plt.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save and show the plot
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ML_RFvsXGB.png'))
plt.show()
