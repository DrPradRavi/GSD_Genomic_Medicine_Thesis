import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import multiprocessing
import os
from itertools import product
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


PARAM_GRID = {
    'n_estimators': [300, 500,1000],
    'max_depth': [3, 5, 10], 
    'max_features': [100, 300, 500] 
}
FINAL_MODEL_REPEATS = 100
CV_N_SPLITS = 5
CV_N_REPEATS = 5
NUM_FEATURES_TO_SELECT = 1000
TARGET_COLUMN = 'Case1_Control0'

# Set up output directory
output_dir = 'FS_RF'
os.makedirs(output_dir, exist_ok=True)

def evaluate_params(params, X, y):
    rf = RandomForestClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)
    accuracy_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    print(f"Completed evaluation for params: {params}")
    return params, np.mean(accuracy_scores), np.mean(auc_scores)

def run_final_model(best_params, X, y):
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
    
    n, bins, patches = plt.hist(importance_df['importance'], bins=20, 
                                facecolor='none', edgecolor='black', linewidth=1)
    
    for i in range(len(n)):
        center = (bins[i] + bins[i+1]) / 2
        plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom')
    
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
    plt.savefig(os.path.join(output_dir, 'FS_RF_feature_importance_histogram.png'))
    plt.close()
    print("Feature importance histogram plotted and saved.")

def plot_p_value_histogram(importance_df, output_dir):
    print("Plotting p-value histogram...")
    
    plt.figure(figsize=(12, 6))
    
    n, bins, patches = plt.hist(importance_df['p_value'], bins=20, 
                                facecolor='none', edgecolor='black', linewidth=1)
    
    for i in range(len(n)):
        center = (bins[i] + bins[i+1]) / 2
        plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom')
    
    mean_p_value = importance_df['p_value'].mean()
    median_p_value = importance_df['p_value'].median()
    
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
    plt.savefig(os.path.join(output_dir, 'FS_RF_p_value_histogram.png'))
    plt.close()
    print("p-value histogram plotted and saved.")

def load_large_csv(file_path):
    print("Loading dataset...")
    try:
        chunks = pd.read_csv(file_path, 
                             encoding='utf-8-sig', 
                             on_bad_lines='warn',   
                             low_memory=False,
                             chunksize=10000)
        
        df = pd.concat(chunks, ignore_index=True)
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("First 5 values of the index:")
        print(df.iloc[:5, 0].tolist())
        
        return df
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
    
    return None

def main():
    print("Starting Random Forest Feature Selection process...")

    print("Loading data...")
    data = load_large_csv('Part8d_output.csv')
    
    if data is None:
        raise ValueError("Failed to load dataset")

    data.set_index(data.columns[0], inplace=True)
    TARGET_COLUMN = data.columns[-1]

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print(f"Dataset loaded. Shape: {X.shape}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Number of positive cases: {sum(y == 1)}")
    print(f"Number of negative cases: {sum(y == 0)}")

    print("\nStarting parameter optimization...")
    num_cores_tuning = min(4, multiprocessing.cpu_count())
    results = Parallel(n_jobs=num_cores_tuning)(
        delayed(evaluate_params)(dict(zip(PARAM_GRID.keys(), v)), X, y)
        for v in product(*PARAM_GRID.values())
    )
    print("Parameter optimization completed.")

    print("\nSaving parameter configurations and scores...")
    config_results = [{**params, 'avg_accuracy': accuracy, 'avg_auc': auc} for params, accuracy, auc in results]
    config_df = pd.DataFrame(config_results)
    config_df.to_csv(os.path.join(output_dir, 'FS_RF_parameter_configurations.csv'), index=False)
    print("Parameter configurations saved.")

    print("\nSelecting best parameters...")
    best_params, best_accuracy, best_auc = max(results, key=lambda x: x[1])
    pd.DataFrame([best_params]).to_csv(os.path.join(output_dir, 'FS_RF_best_params.csv'), index=False)
    print(f"Best parameters saved. Best accuracy: {best_accuracy}")
    print(f"Best performing parameter configuration:")
    print(f"Parameters: {best_params}")
    print(f"Accuracy: {best_accuracy}")
    print(f"AUC: {best_auc}")

    print(f"\nRunning final model {FINAL_MODEL_REPEATS} times...")
    num_cores_final = min(2, multiprocessing.cpu_count())
    all_importances = []
    for i in range(FINAL_MODEL_REPEATS):
        try:
            importance = run_final_model(best_params, X, y)
            all_importances.append(importance)
            print(f"Completed run {i+1}/{FINAL_MODEL_REPEATS}")
        except Exception as e:
            print(f"Error in run {i+1}: {str(e)}")
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
    importance_df.to_csv(os.path.join(output_dir, 'FS_RF_all_features_importance.csv'), index=False)

    print("\nPlotting p-value histogram...")
    plot_p_value_histogram(importance_df, output_dir)

    print("\nSelecting top features...")
    importance_df = importance_df.sort_values('importance', ascending=False)
    top_features = importance_df.head(NUM_FEATURES_TO_SELECT).copy()
    top_features['rank'] = range(1, len(top_features) + 1)

    print("Saving top features...")
    top_features.to_csv(os.path.join(output_dir, 'FS_RF_top_features.csv'), index=False)

    print("\nSaving selected features dataset...")
    selected_features = X[top_features['feature']].copy()
    selected_features[TARGET_COLUMN] = y
    selected_features.to_csv(os.path.join(output_dir, 'FS_RF_selected_features_dataset.csv'), index=True)

    print("\nPlotting feature importance histogram...")
    plot_feature_importance_histogram(importance_df, output_dir)

    print("\nFeature selection complete. Results saved in", output_dir)
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {NUM_FEATURES_TO_SELECT}")

if __name__ == '__main__':
    main()