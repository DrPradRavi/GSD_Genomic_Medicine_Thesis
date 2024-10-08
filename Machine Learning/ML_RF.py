import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from scipy import stats
import multiprocessing
import os
from itertools import product
import matplotlib.pyplot as plt

PARAM_GRID = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [3, 5, 10], 
    'max_features': [50, 100, 150, 200],
    'min_samples_leaf': [1, 2, 4] 
}
FINAL_MODEL_REPEATS = 100
CV_N_SPLITS = 10
CV_N_REPEATS = 10
NUM_FEATURES_TO_SELECT = 40
TARGET_COLUMN = 'Case1_Control0'

output_dir = 'ML_RF'
os.makedirs(output_dir, exist_ok=True)

def evaluate_params(args):
    params, X, y = args
    rf = RandomForestClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)
    accuracy_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    print(f"Completed evaluation for params: {params}")
    return params, np.mean(accuracy_scores), np.mean(auc_scores)

def run_final_model(args):
    best_params, X, y = args
    rf = RandomForestClassifier(**best_params)
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)
    importances = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf.fit(X_train, y_train)
        importances.append(rf.feature_importances_)
    return np.mean(importances, axis=0)

def plot_feature_importance_histogram(importance_df, output_dir):
    print("Plotting feature importance histogram...")
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(importance_df['importance'], bins=75, 
                                facecolor='none', edgecolor='black', linewidth=1)
    for i in range(len(n)):
        if n[i] > 0:
            center = (bins[i] + bins[i+1]) / 2
            plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom', fontsize=8)
    
    mean_importance = importance_df['importance'].mean()
    median_importance = importance_df['importance'].median()
    threshold = importance_df['importance'].iloc[NUM_FEATURES_TO_SELECT - 1]
    
    plt.axvline(mean_importance, color='r', linestyle='--', label=f'Mean: {mean_importance:.6f}')
    plt.axvline(median_importance, color='b', linestyle='-', label=f'Median: {median_importance:.6f}')
    plt.axvline(threshold, color='g', linestyle=':', label=f'Threshold ({NUM_FEATURES_TO_SELECT}th feature): {threshold:.6f}')
    
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Importance Scores')
    plt.legend()
    max_importance = importance_df['importance'].max()
    plt.xlim(0, max_importance * 1.1)
    plt.ylim(top=plt.ylim()[1] * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ML_RF_feature_importance_histogram.png'), dpi=300)
    plt.close()
    print("Feature importance histogram plotted and saved.")

def plot_p_value_histogram(importance_df, output_dir):
    print("Plotting p-value histogram...")
    plt.figure(figsize=(12, 6))
    bins = np.logspace(np.log10(importance_df['p_value'].min()), np.log10(1), 75)
    n, bins, patches = plt.hist(importance_df['p_value'], bins=bins, 
                                facecolor='none', edgecolor='black', linewidth=1)
    
    # Add frequency labels above each bar, centered and upright
    for i in range(len(n)):
        if n[i] > 0:
            center = np.sqrt(bins[i] * bins[i+1])
            plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom', fontsize=8)
    
    mean_p_value = importance_df['p_value'].mean()
    median_p_value = importance_df['p_value'].median()
    
    plt.axvline(mean_p_value, color='r', linestyle='--', label=f'Mean: {mean_p_value:.2e}')
    plt.axvline(median_p_value, color='b', linestyle='-', label=f'Median: {median_p_value:.2e}')
    plt.axvline(0.05, color='g', linestyle=':', label='p = 0.05')
    
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values')
    plt.legend()
    
    plt.xscale('log')
    plt.xlim(importance_df['p_value'].min() * 0.9, 1)
    plt.ylim(top=plt.ylim()[1] * 1.1)
    
    # Customize x-axis ticks
    tick_locations = [1e-50, 1e-40, 1e-30, 1e-20, 1e-10, 1e-5, 0.001, 0.01, 0.05, 0.1, 0.5, 1]
    plt.xticks(tick_locations, [f'10^{int(np.log10(x))}' if x < 1e-3 else f'{x:.3f}' for x in tick_locations])
    
    plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels
    plt.gca().tick_params(axis='x', which='minor', length=0)  # Hide minor ticks
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ML_RF_p_value_histogram.png'), dpi=300)
    plt.close()
    print("p-value histogram plotted and saved.")

def main():
    print("Starting Random Forest Machine Learning process...")
    print("Loading data...")
    data = pd.read_csv('FS_Merge_Output/merged_features.csv', index_col=0)
    
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    print(f"Dataset loaded. Shape: {X.shape}")
    print(f"Number of positive cases: {sum(y == 1)}")
    print(f"Number of negative cases: {sum(y == 0)}")

    print("\nStarting parameter optimization...")
    num_cores = max(1, multiprocessing.cpu_count() - 6)
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(evaluate_params, [(dict(zip(PARAM_GRID.keys(), v)), X, y) for v in product(*PARAM_GRID.values())])
    print("Parameter optimization completed.")

    print("\nSaving parameter configurations and scores...")
    config_results = [
        {**params, 'avg_accuracy': accuracy, 'avg_auc': auc}
        for params, accuracy, auc in results
    ]
    config_df = pd.DataFrame(config_results)
    config_df.to_csv(os.path.join(output_dir, 'ML_RF_parameter_configurations.csv'), index=False)
    print("Parameter configurations saved.")

    print("\nSelecting best parameters...")
    best_params, best_accuracy, _ = max(results, key=lambda x: x[1])
    pd.DataFrame([best_params]).to_csv(os.path.join(output_dir, 'ML_RF_best_params.csv'), index=False)
    print(f"Best parameters saved. Best accuracy: {best_accuracy}")

    print(f"\nRunning final model {FINAL_MODEL_REPEATS} times...")
    with multiprocessing.Pool(num_cores) as pool:
        all_importances = pool.map(run_final_model, [(best_params, X, y) for _ in range(FINAL_MODEL_REPEATS)])
    print("Final model runs completed.")

    print("\nCalculating feature importances and p-values...")
    mean_importances = np.mean(all_importances, axis=0)
    p_values = [stats.ttest_1samp([imp[i] for imp in all_importances], 0.0)[1] for i in range(len(mean_importances))]

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_importances,
        'p_value': p_values
    })

    print("Saving all features importance scores...")
    importance_df.to_csv(os.path.join(output_dir, 'ML_RF_all_features_importance.csv'), index=False)

    print("\nPlotting p-value histogram...")
    plot_p_value_histogram(importance_df, output_dir)

    print("\nSelecting top features...")
    importance_df = importance_df.sort_values('importance', ascending=False)
    top_features = importance_df.head(NUM_FEATURES_TO_SELECT).copy()
    top_features['rank'] = range(1, len(top_features) + 1)

    print("Saving top features...")
    top_features.to_csv(os.path.join(output_dir, 'ML_RF_top_features.csv'), index=False)

    print("\nSaving selected features dataset...")
    selected_features = X[top_features['feature']].copy()
    selected_features[TARGET_COLUMN] = y
    selected_features.to_csv(os.path.join(output_dir, 'ML_RF_selected_features_dataset.csv'), index=True)

    print("\nPlotting feature importance histogram...")
    plot_feature_importance_histogram(importance_df, output_dir)

    print("\nRandom Forest Machine Learning process complete. Results saved in", output_dir)
    print(f"Total features: {X.shape[1]}")
    print(f"Selected features: {NUM_FEATURES_TO_SELECT}")

if __name__ == '__main__':
    main()