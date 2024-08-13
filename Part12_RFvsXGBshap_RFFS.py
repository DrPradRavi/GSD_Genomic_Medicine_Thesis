import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants
OUTPUT_DIR = "Part12_Comparison_output_RF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
XGB_importance_df = pd.read_csv("Part11_XGB_output_RF/Part11_XGB_all_features_importance.csv")
rf_importance_df = pd.read_csv("Part10_RF_output_RF/Part10_RF_all_features_importance.csv")

# Merge data on feature
merged_df = pd.merge(XGB_importance_df, rf_importance_df, on='feature', suffixes=('_XGB', '_rf'))

# Load top features
XGB_top_features = pd.read_csv("Part11_XGB_output_RF/Part11_XGB_top_features_importance.csv")
rf_top_features = pd.read_csv("Part10_RF_output_RF/Part10_RF_top_features_importance.csv")

# Create category columns
merged_df['category'] = "Wasn't selected"
merged_df.loc[merged_df['feature'].isin(rf_top_features['feature']), 'category'] = 'RF selected'
merged_df.loc[merged_df['feature'].isin(XGB_top_features['feature']), 'category'] = 'XGB selected'
merged_df.loc[merged_df['feature'].isin(rf_top_features['feature']) & 
              merged_df['feature'].isin(XGB_top_features['feature']), 'category'] = 'RF and XGB selected'

# Set up the plot
plt.figure(figsize=(12, 10))
plt.grid(True)

# Define categories, colors, markers, sizes, and alphas
categories = ["Wasn't selected", 'RF selected', 'XGB selected', 'RF and XGB selected']
colors = ['grey', 'red', 'blue', 'purple']
markers = ['o', '+', 'x', '*']
sizes = [20, 100, 100, 250]
alphas = [0.4, 0.6, 0.5, 1.0]

# Create the scatter plot
for cat, color, marker, size, alpha in zip(categories, colors, markers, sizes, alphas):
    subset = merged_df[merged_df['category'] == cat]
    if cat in ['RF selected', 'XGB selected']:
        plt.scatter(subset['importance_rf'], subset['importance_XGB'], 
                    c=color, marker=marker, s=size, 
                    alpha=alpha, label=cat, linewidth=2)
    elif cat == 'RF and XGB selected':
        plt.scatter(subset['importance_rf'], subset['importance_XGB'], 
                    facecolors=color, edgecolors='black', marker=marker, s=size, 
                    alpha=alpha, label=cat, linewidth=1)
    else:
        plt.scatter(subset['importance_rf'], subset['importance_XGB'], 
                    c=color, marker=marker, s=size, alpha=alpha, label=cat)

# Customize the plot
plt.xlabel('RF Importance Score')
plt.ylabel('XGB Importance Score')
plt.title('Feature Importance Comparison')
plt.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save and show the plot
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Part12_RFvsXGB.png'))
plt.show()