import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from scipy import stats
from joblib import Parallel, delayed
import os
from itertools import product
import time

# Configuration variables
PARAM_GRID = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.001, 0.01, 0.1],
    'min_child_weight': [1, 5, 10],
    'early_stopping_rounds': [15]
}
FINAL_MODEL_REPEATS = 100
CV_N_SPLITS = 10
CV_N_REPEATS = 10
NUM_FEATURES_TO_SELECT = 40
TARGET_COLUMN = 'Case1_Control0'
output_dir = 'ML_XGB'
os.makedirs(output_dir, exist_ok=True)

def evaluate_params(args):
    index, total, params, X, y = args
    cv_params = params.copy()
    cv_params.pop('early_stopping_rounds', None)
    xgb = XGBClassifier(**cv_params, colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,
                        subsample=0.5, eval_metric='error', 
                        tree_method='hist', grow_policy='lossguide', max_bin=256)
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)
    accuracy_scores = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(xgb, X, y, cv=cv, scoring='roc_auc')
    print(f"Completed evaluation for params: {params}")
    return params, np.mean(accuracy_scores), np.mean(auc_scores)

def run_final_model(args):
    index, best_params, X, y = args
    print(f"Starting final model repeat {index + 1}/{FINAL_MODEL_REPEATS}")
    early_stopping_rounds = best_params.pop('early_stopping_rounds', 15)
    xgb = XGBClassifier(**best_params, colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,
                        subsample=0.5, eval_metric='error', 
                        tree_method='hist', grow_policy='lossguide', max_bin=256)
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)
    scores = cross_val_score(xgb, X, y, cv=cv, scoring='roc_auc')
    xgb.fit(X, y)
    feature_importance = xgb.get_booster().get_score(importance_type='gain')
    importance_values = np.zeros(X.shape[1])
    for feature, importance in feature_importance.items():
        try:
            if feature.startswith('f'):
                feature_index = int(feature[1:])
            else:
                feature_index = int(feature)
            if feature_index < X.shape[1]:
                importance_values[feature_index] = importance
        except ValueError:
            print(f"Warning: Unexpected feature name format: {feature}")
            continue
    print(f"Finished final model repeat {index + 1}/{FINAL_MODEL_REPEATS}")
    return importance_values, np.mean(scores)

def plot_feature_importance_histogram(importance_df, output_dir):
    print("Plotting feature importance histogram...")
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(importance_df['importance'], bins=100, 
                                facecolor='none', edgecolor='black', linewidth=1)
    for i in range(len(n)):
        if n[i] > 0:
            center = (bins[i] + bins[i+1]) / 2
            plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom')
    mean_importance = importance_df['importance'].mean()
    median_importance = importance_df['importance'].median()
    threshold = importance_df['importance'].iloc[NUM_FEATURES_TO_SELECT - 1]
    plt.axvline(mean_importance, color='r', linestyle='--', label=f'Mean: {mean_importance:.6f}')
    plt.axvline(median_importance, color='b', linestyle='-', label=f'Median: {median_importance:.6f}')
    plt.axvline(threshold, color='g', linestyle=':', label=f'Threshold ({NUM_FEATURES_TO_SELECT}th feature): {threshold:.6f}')
    plt.xlabel('Feature Importance Score (Gain)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Importance Scores')
    plt.legend()
    plt.xlim(0, importance_df['importance'].max() * 1.1)
    plt.ylim(top=plt.ylim()[1] * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ML_XGB_feature_importance_histogram.png'))
    plt.close()
    print("Feature importance histogram plotted and saved.")

def plot_p_value_histogram(importance_df, output_dir):
    print("Plotting p-value histogram...")
    plt.figure(figsize=(12, 6))
    valid_p_values = importance_df['p_value'][(~np.isnan(importance_df['p_value'])) & (importance_df['p_value'] < 1)]
    if len(valid_p_values) == 0:
        print("No valid p-values to plot. Skipping p-value histogram.")
        return
    n, bins, patches = plt.hist(valid_p_values, bins=100, 
                                facecolor='none', edgecolor='black', linewidth=1)
    for i in range(len(n)):
        if n[i] > 0:
            center = (bins[i] + bins[i+1]) / 2
            plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom')
    mean_p_value = np.mean(valid_p_values)
    median_p_value = np.median(valid_p_values)
    plt.axvline(mean_p_value, color='r', linestyle='--', label=f'Mean: {mean_p_value:.6f}')
    plt.axvline(median_p_value, color='b', linestyle='-', label=f'Median: {median_p_value:.6f}')
    plt.axvline(0.05, color='g', linestyle=':', label='p = 0.05')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(top=plt.ylim()[1] * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ML_XGB_p_value_histogram.png'))
    plt.close()
    print("p-value histogram plotted and saved.")

def main():
    print("Starting XGBoost Machine Learning process...")
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
    num_cores = max(1, os.cpu_count() - 6)
    param_combinations = list(product(*PARAM_GRID.values()))
    total_combinations = len(param_combinations)
    results = Parallel(n_jobs=num_cores)(
        delayed(evaluate_params)((i, total_combinations, dict(zip(PARAM_GRID.keys(), v)), X, y)) 
        for i, v in enumerate(param_combinations)
    )
    print("Parameter optimization completed.")

    print("\nSaving parameter configurations and scores...")
    config_results = [{**params, 'avg_accuracy': accuracy, 'avg_auc': auc} for params, accuracy, auc in results]
    config_df = pd.DataFrame(config_results)
    config_df.to_csv(os.path.join(output_dir, 'ML_XGB_parameter_configurations.csv'), index=False)
    print("Parameter configurations saved.")

    print("\nSelecting best parameters...")
    best_params, best_accuracy, _ = max(results, key=lambda x: x[1])
    pd.DataFrame([best_params]).to_csv(os.path.join(output_dir, 'ML_XGB_best_params.csv'), index=False)
    print(f"Best parameters saved. Best accuracy: {best_accuracy}")

    print(f"\nRunning final model {FINAL_MODEL_REPEATS} times...")
    start_time = time.time()
    results = Parallel(n_jobs=num_cores)(
        delayed(run_final_model)((i, best_params, X, y)) for i in range(FINAL_MODEL_REPEATS)
    )
    all_importances, all_scores = zip(*results)
    print("Final model runs completed.")
    print(f"Average ROC AUC score: {np.mean(all_scores):.4f} (±{np.std(all_scores):.4f})")

    print("\nCalculating feature importances and p-values...")
    mean_importances = np.mean(all_importances, axis=0)
    p_values = [stats.ttest_1samp([imp[i] for imp in all_importances], 0.0)[1] for i in range(len(mean_importances))]
    print("Feature importances and p-values calculated.")

    print("Saving all features importance scores...")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_importances,
        'p_value': p_values
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df.to_csv(os.path.join(output_dir, 'ML_XGB_all_features_importance.csv'), index=False)
    print("All features importance scores saved.")

    print("\nPlotting p-value histogram...")
    plot_p_value_histogram(importance_df, output_dir)

    print("\nSelecting top features...")
    top_features = importance_df.head(NUM_FEATURES_TO_SELECT).copy()

    print("Saving top features...")
    top_features.to_csv(os.path.join(output_dir, 'ML_XGB_top_features.csv'), index=False)

    print("\nSaving selected features dataset...")
    selected_features = X[top_features['feature']].copy()
    selected_features[TARGET_COLUMN] = y
    selected_features.to_csv(os.path.join(output_dir, 'ML_XGB_selected_features_dataset.csv'), index=True)

    print("\nPlotting feature importance histogram...")
    plot_feature_importance_histogram(importance_df, output_dir)

    end_time = time.time()
    print(f"\nXGBoost Machine Learning process complete. Results saved in {output_dir}")
    print(f"Total features: {X.shape[1]}")
    print(f"Selected features: {NUM_FEATURES_TO_SELECT}")
    print(f"Process completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()