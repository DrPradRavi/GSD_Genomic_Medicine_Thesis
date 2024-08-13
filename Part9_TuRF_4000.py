import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skrebate import TuRF
from scipy.stats import chi2_contingency
import time
import csv
import psutil  # Add this import

INPUT_FILE = 'Part8b_output.csv'
OUTPUT_DIR = 'Part9_TuRF_feature_selection_results'
TURF_FILE = f'{OUTPUT_DIR}/Part9_turf_selected.csv'
SELECTED_FEATURES_FILE = f'{OUTPUT_DIR}/Part9_selected_features.csv'
PERFORMANCE_FILE = f'{OUTPUT_DIR}/performance_metrics.csv'

# Configuration variables
N_NEIGHBORS = 10
PCT_FEATURES_REMOVED = 0.1
N_ITERATIONS = 10
MIN_FEATURES_TO_SELECT = 1000
MAX_FEATURES_TO_SELECT = 4000
IMPORTANCE_THRESHOLD = 0.001

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

def turf_feature_importance(X, y, n_neighbors=N_NEIGHBORS, pct_features_removed=PCT_FEATURES_REMOVED, n_iterations=N_ITERATIONS):
    print(f"Running TuRF feature selection with {n_iterations} iterations...")
    
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    encoder = OrdinalEncoder(categories=[[0, 1, 2]])
    
    print("Encoding features...")
    for column in tqdm(X_imputed.columns, desc="Encoding"):
        X_imputed[column] = encoder.fit_transform(X_imputed[column].values.reshape(-1, 1))
    
    n_features_to_select = min(MAX_FEATURES_TO_SELECT, X.shape[1])
    
    # Calculate batch size based on available memory
    total_memory = 24 * 1024 * 1024 * 1024  # 24GB in bytes
    safe_memory = total_memory * 0.7  # Using 70% of available memory
    estimated_feature_size = 8  # bytes per feature
    num_samples = X.shape[0]
    
    batch_size = int(safe_memory / (num_samples * estimated_feature_size))
    print(f"Calculated batch size: {batch_size}")

    # Process data in batches
    feature_importances = []
    
    for i in range(0, X.shape[1], batch_size):
        print(f"Processing batch {i//batch_size + 1}")
        X_batch = X_imputed.iloc[:, i:i+batch_size]
        
        turf = TuRF(core_algorithm="ReliefF", 
                    n_features_to_select=min(n_features_to_select, X_batch.shape[1]), 
                    n_neighbors=n_neighbors, 
                    pct=pct_features_removed, 
                    n_jobs=-1, 
                    verbose=True)
        
        X_array = X_batch.values
        headers = X_batch.columns.tolist()
        
        print("Fitting TuRF...")
        turf.fit(X_array, y, headers)
        
        feature_importances.extend(list(zip(X_batch.columns, turf.feature_importances_)))
    
    feature_importance_df = pd.DataFrame(feature_importances, columns=['feature', 'importance'])
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Select features based on importance threshold and desired range
    selected_features = feature_importance_df[feature_importance_df['importance'] > IMPORTANCE_THRESHOLD]
    if len(selected_features) < MIN_FEATURES_TO_SELECT:
        selected_features = feature_importance_df.head(MIN_FEATURES_TO_SELECT)
    elif len(selected_features) > MAX_FEATURES_TO_SELECT:
        selected_features = feature_importance_df.head(MAX_FEATURES_TO_SELECT)
    
    selected_features['rank'] = range(1, len(selected_features) + 1)
    
    print(f"TuRF feature selection completed. Selected {len(selected_features)} features.")
    return feature_importance_df, selected_features

def plot_feature_importance_histogram(feature_importance_df, selected_features, output_dir):
    print("Plotting feature importance histogram...")
    
    plt.figure(figsize=(12, 6))
    
    # Use the full range of importance scores
    n, bins, patches = plt.hist(feature_importance_df['importance'], bins=20, 
                                facecolor='none', edgecolor='black', linewidth=1)
    
    for i in range(len(n)):
        center = (bins[i] + bins[i+1]) / 2
        plt.text(center, n[i], str(int(n[i])), ha='center', va='bottom')
    
    mean_importance = feature_importance_df['importance'].mean()
    median_importance = feature_importance_df['importance'].median()
    threshold = selected_features['importance'].min()
    
    print(f"Mean: {mean_importance:.6f}")
    print(f"Median: {median_importance:.6f}")
    print(f"Threshold: {threshold:.6f}")
    
    plt.axvline(mean_importance, color='r', linestyle='--', label=f'Mean: {mean_importance:.6f}')
    plt.axvline(median_importance, color='b', linestyle='-', label=f'Median: {median_importance:.6f}')
    plt.axvline(threshold, color='g', linestyle=':', label=f'Threshold ({len(selected_features)} features): {threshold:.6f}')
    
    plt.xlabel('SNP Importance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of SNP Importance Scores (TuRF)')
    plt.legend()
    
    # Set x-axis limits to include the full range of scores
    min_importance = feature_importance_df['importance'].min()
    max_importance = feature_importance_df['importance'].max()
    
    # Set x-axis lower limit to 0 if all values are positive, otherwise to min_importance * 1.1
    x_min = 0 if min_importance >= 0 else min_importance * 1.1
    plt.xlim(x_min, max_importance * 1.1)
    
    plt.ylim(top=plt.ylim()[1] * 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Part9_turf_importance_histogram.png')
    plt.close()
    print("Feature importance histogram plotted and saved.")

def calculate_p_values(X, y, selected_features):
    print("Calculating p-values...")
    p_values = []
    for feature in tqdm(selected_features['feature'], desc="Calculating p-values"):
        contingency_table = pd.crosstab(X[feature], y)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        p_values.append(p_value)
    
    selected_features['p_value'] = p_values
    selected_features['bonferroni_p_value'] = np.minimum(selected_features['p_value'] * len(selected_features), 1)
    
    selected_features = selected_features.sort_values('p_value')
    print("P-values calculated and added to the selected features DataFrame.")
    return selected_features

def plot_p_value_distribution(selected_features, output_dir):
    print("Plotting p-value distribution...")
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(selected_features['p_value'], bins=50, edgecolor='black')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of P-values for Selected Features')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Part9_p_value_distribution.png')
    plt.close()
    print("P-value distribution plotted and saved.")

def main():
    print("Starting main process...")
    start_time = time.time()
    max_ram_usage = 0  # Initialize max RAM usage tracker
    
    try:
        print("Step 1/6: Loading dataset...")
        df = load_large_csv(INPUT_FILE)
        if df is None:
            raise ValueError("Failed to load dataset")
        
        print("Step 2/6: Preprocessing data...")
        df = preprocess_data(df)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(int).values
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("Step 3/6: Running TuRF feature selection...")
        feature_importance_df, selected_features = turf_feature_importance(X, y)
        
        print("Step 4/6: Calculating p-values...")
        selected_features = calculate_p_values(X, y, selected_features)
        
        print("Step 5/6: Saving results and plotting...")
        print(f"Saving selected features to {TURF_FILE}...")
        selected_features_sorted = selected_features.sort_values(by='rank')
        selected_features_sorted.to_csv(TURF_FILE, index=False)
        print(f"Selected features saved to {TURF_FILE}.")
        plot_feature_importance_histogram(feature_importance_df, selected_features, OUTPUT_DIR)
        plot_p_value_distribution(selected_features, OUTPUT_DIR)
        
        print(f"Step 6/6: Selecting {len(selected_features)} features and saving to CSV...")
        selected_feature_names = selected_features['feature'].tolist()
        selected_columns = sorted(selected_feature_names + ['Case1_Control0'])
        selected_df = df[selected_columns]
        selected_df.to_csv(SELECTED_FEATURES_FILE, index=True)
        print(f"Selected features saved to {SELECTED_FEATURES_FILE}.")
        
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