import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import psutil
import csv

INPUT_FILE = 'Part8b_output.csv'
OUTPUT_DIR = 'Part9_RF_feature_selection_results'
RF_FILE = f'{OUTPUT_DIR}/Part9_rf_selected.csv'
SELECTED_FEATURES_FILE = f'{OUTPUT_DIR}/Part9_selected_features.csv'
BONFERRONI_FILE = f'{OUTPUT_DIR}/Part9_rf_selected_Bonferroni.csv'
SELECTED_FEATURES_BONFERRONI_FILE = f'{OUTPUT_DIR}/Part9_selected_features_Bonferroni.csv'
PERFORMANCE_FILE = f'{OUTPUT_DIR}/performance_metrics.csv'

# Configuration variables
N_GINI_ITERATIONS = 100
N_PERMUTATIONS = 100
N_PARAM_PERMUTATIONS = 10

# New constants for feature selection
MIN_FEATURES = 1000
MAX_FEATURES = 4000
IMPORTANCE_THRESHOLD = 0.001  # Adjust this value based on your data

def load_large_csv(file_path):
    print("Loading dataset...")
    try:
        # Verify the file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"The file {file_path} is empty.")
        
        # Print file details
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print("First few lines of the file:")
        with open(file_path, 'r') as f:
            for _ in range(5):
                print(f.readline().strip())
        
        # Use pandas to read the CSV file
        df = pd.read_csv(file_path, 
                         encoding='utf-8-sig',  # Handle potential BOM
                         on_bad_lines='warn',   # Warn about problematic lines
                         low_memory=False,      # Disable low memory warnings
                         index_col=0)           # Use the first column as index
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("First 5 values of the index:")
        print(df.index[:5].tolist())
        
        return df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Error with file content: {e}")
    except UnicodeDecodeError:
        print("Error decoding the file. Try specifying a different encoding.")
    except pd.errors.EmptyDataError:
        print("The CSV file is empty or contains no valid data.")
    except MemoryError:
        print("Not enough memory to load the entire dataset. Consider using chunking or other memory-saving techniques.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
    
    return None

def preprocess_data(df):
    print("Preprocessing data...")
    df.replace(-1, pd.NA, inplace=True)
    print("Data preprocessed.")
    return df

def optimize_rf_parameters(X, y):
    print("Optimizing Random Forest parameters...")
    param_grid = {
        'n_estimators': [10, 50, 100, 150],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_depth': [5, 10, 20, 30]
    }
    best_params = None
    best_score = -np.inf
    
    for params in ParameterGrid(param_grid):
        scores = []
        for _ in range(N_PARAM_PERMUTATIONS):
            rf = RandomForestClassifier(**params, max_features='sqrt', random_state=np.random.randint(0, 10000))
            rf.fit(X, y)
            y_pred = rf.predict_proba(X)[:, 1]
            scores.append(roc_auc_score(y, y_pred))
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print(f"Best parameters found: {best_params} with average AUC: {best_score}")
    return best_params

def random_forest_feature_importance(X, y, best_params, max_features='sqrt', n_gini_iterations=N_GINI_ITERATIONS, n_permutations=N_PERMUTATIONS):
    print(f"Training Random Forest and computing average Gini importance over {n_gini_iterations} iterations...")
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    encoder = OrdinalEncoder(categories=[[0, 1, 2]])
    
    for column in X_imputed.columns:
        X_imputed[column] = encoder.fit_transform(X_imputed[column].values.reshape(-1, 1))
    
    importances = np.zeros(X.shape[1])
    for i in range(n_gini_iterations):
        rf = RandomForestClassifier(**best_params, max_features=max_features, random_state=np.random.randint(0, 10000))
        rf.fit(X_imputed, y)
        importances += rf.feature_importances_
    importances /= n_gini_iterations
    
    print(f"Computing p-values using {n_permutations} permutations...")
    null_importances = np.zeros((n_permutations, X.shape[1]))
    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)
        rf = RandomForestClassifier(**best_params, max_features=max_features, random_state=np.random.randint(0, 10000))
        rf.fit(X_imputed, y_permuted)
        null_importances[i] = rf.feature_importances_
    
    p_values = np.mean(null_importances >= importances, axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances,
        'p_value': p_values
    }).sort_values('importance', ascending=False)
    feature_importance_df['p_value'] = feature_importance_df['p_value'].apply(lambda x: f"{x:.9f}")
    feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()
    
    # Select features based on importance threshold and desired range
    selected_features = feature_importance_df[feature_importance_df['importance'] > IMPORTANCE_THRESHOLD]
    if len(selected_features) < MIN_FEATURES:
        selected_features = feature_importance_df.head(MIN_FEATURES)
    elif len(selected_features) > MAX_FEATURES:
        selected_features = feature_importance_df.head(MAX_FEATURES)
    
    print(f"Random Forest training and average Gini importance computation completed. Selected {len(selected_features)} features.")
    return feature_importance_df, selected_features

def plot_feature_importance_histogram(feature_importance_df, output_dir):
    print("Plotting feature importance histogram...")
    
    plt.figure(figsize=(12, 6))
    
    # Create the histogram with unfilled bars and get the histogram data
    n, bins, patches = plt.hist(feature_importance_df['importance'], bins=20, 
                                facecolor='none', edgecolor='black', linewidth=1)
    
    # Add frequency labels above each bar, centered
    for i in range(len(n)):
        center = (bins[i] + bins[i+1]) / 2
        plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom')
    
    mean_importance = feature_importance_df['importance'].mean()
    median_importance = feature_importance_df['importance'].median()
    threshold = feature_importance_df['importance'].iloc[MIN_FEATURES - 1]
    
    # Print out the values for debugging
    print(f"Mean: {mean_importance:.6f}")
    print(f"Median: {median_importance:.6f}")
    print(f"Threshold: {threshold:.6f}")
    
    plt.axvline(mean_importance, color='r', linestyle='--', label=f'Mean: {mean_importance:.6f}')
    plt.axvline(median_importance, color='b', linestyle='-', label=f'Median: {median_importance:.6f}')
    plt.axvline(threshold, color='g', linestyle=':', label=f'Threshold ({MIN_FEATURES}th feature): {threshold:.6f}')
    
    plt.xlabel('SNP Importance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of SNP Importance Scores')
    plt.legend()
    
    # Set x-axis to start at 0 and end at a value slightly larger than the maximum importance
    max_importance = feature_importance_df['importance'].max()
    plt.xlim(0, max_importance * 1.1)
    
    # Adjust y-axis limit to accommodate the frequency labels
    plt.ylim(top=plt.ylim()[1] * 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Part9_rf_importance_histogram.png')
    plt.close()
    print("Feature importance histogram plotted and saved.")

def main():
    print("Starting main process...")
    start_time = time.time()
    max_ram_usage = 0
    
    try:
        print("Step 1/7: Loading dataset...")
        df = load_large_csv(INPUT_FILE)
        if df is None:
            raise ValueError("Failed to load dataset")
        
        print("Step 2/7: Preprocessing data...")
        df = preprocess_data(df)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(int).values
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("Step 3/7: Optimizing Random Forest parameters...")
        best_params = optimize_rf_parameters(X, y)
        
        print("Step 4/7: Computing feature importance...")
        feature_importance_df, selected_features = random_forest_feature_importance(X, y, best_params)
        
        print("Step 5/7: Saving results and plotting...")
        print(f"Saving selected features to {RF_FILE}...")
        selected_features_sorted = selected_features.sort_values(by='rank')
        selected_features_sorted.loc[:, 'p_value'] = selected_features_sorted['p_value'].apply(lambda x: f"{float(x):.9f}")
        selected_features_sorted.to_csv(RF_FILE, index=False)
        print(f"Selected features saved to {RF_FILE}.")
        plot_feature_importance_histogram(feature_importance_df, OUTPUT_DIR)
        
        print(f"Step 6/7: Selecting {len(selected_features)} features and saving to CSV...")
        selected_feature_names = selected_features['feature'].tolist()
        selected_columns = sorted(selected_feature_names + ['Case1_Control0'])
        selected_df = df[selected_columns]
        selected_df.to_csv(SELECTED_FEATURES_FILE, index=True)
        print(f"Selected features saved to {SELECTED_FEATURES_FILE}.")
        
        print("Step 7/7: Applying Bonferroni correction...")
        n_features = X.shape[1]
        bonferroni_p_value = 0.05 / n_features
        print(f"Bonferroni correction p-value: {bonferroni_p_value:.9f}")
        
        bonferroni_filtered_df = selected_features_sorted[selected_features_sorted['p_value'].astype(float) <= bonferroni_p_value]
        bonferroni_filtered_df.loc[:, 'p_value'] = bonferroni_filtered_df['p_value'].apply(lambda x: f"{float(x):.9f}")
        bonferroni_filtered_df.to_csv(BONFERRONI_FILE, index=False)
        print(f"Bonferroni filtered features saved to {BONFERRONI_FILE}.")
        
        print("Step 7/7: Saving Bonferroni filtered features to CSV...")
        bonferroni_selected_features = bonferroni_filtered_df['feature'].tolist()
        bonferroni_selected_columns = sorted(bonferroni_selected_features + ['Case1_Control0'])
        bonferroni_selected_df = df[bonferroni_selected_columns]
        bonferroni_selected_df.to_csv(SELECTED_FEATURES_BONFERRONI_FILE, index=True)
        print(f"Bonferroni selected features saved to {SELECTED_FEATURES_BONFERRONI_FILE}.")
        
        # Track system usage
        ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
        max_ram_usage = max(max_ram_usage, ram_usage)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save performance metrics
        with open(PERFORMANCE_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Execution Time (s)', 'Max RAM Usage (MB)'])
            writer.writerow([execution_time, max_ram_usage])
        
        print(f"Performance metrics saved to {PERFORMANCE_FILE}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Max RAM usage: {max_ram_usage:.2f} MB")
        print("Main process completed.")

if __name__ == "__main__":
    main()