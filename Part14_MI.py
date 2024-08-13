import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import entropy
from tqdm import tqdm

# Ensure output directory exists
output_dir = 'Part14_outputs'
os.makedirs(output_dir, exist_ok=True)

# Get the number of available cores
available_cores = multiprocessing.cpu_count()

# Set the number of cores to use (6 less than max, but at least 1)
n_jobs = max(1, available_cores - 6)

print(f"Using {n_jobs} cores for parallel processing.")

# Load the data
print("Loading data...")
df = pd.read_csv('Part13_outputs/combined_all_features.csv', index_col=0)
print("Data loaded successfully.")

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Print unique values in each column
print("\nUnique values in each column:")
for col in X.columns:
    unique_values = X[col].unique()
    print(f"{col}: {unique_values}")

# Print unique values in the target variable
print(f"\nTarget variable: {y.name}")
print(f"Unique values: {y.unique()}")

# Check data types
print("\nData types:")
print(X.dtypes)

# Attempt encoding
print("\nAttempting encoding...")
try:
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    print("Encoding successful.")
except Exception as e:
    print(f"Encoding failed. Error: {str(e)}")

def compute_mutual_information_2d(X, y):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    H_X = entropy(np.histogram(X, bins=20)[0])
    H_y = entropy(np.histogram(y, bins=20)[0])
    
    H_Xy = entropy(np.histogram2d(X[:, 0], y, bins=20)[0].flatten())
    
    if X.shape[1] > 1:
        H_Xy += entropy(np.histogram2d(X[:, 1], y, bins=20)[0].flatten())
    
    return H_X + H_y - H_Xy

def rogers_huff_r(x, y):
    epsilon = 1e-8  # Small value to prevent division by zero or negative square root
    p1 = np.mean(x)
    p2 = np.mean(y)
    D = np.mean(x * y) - p1 * p2
    denominator = np.sqrt(max(p1 * (1 - p1) * p2 * (1 - p2), epsilon))
    r = D / denominator if denominator != 0 else 0
    r = np.clip(r, -1, 1)
    return r ** 2

def compute_pairwise_mi_and_ld(X, y):
    n_features = X.shape[1]
    total_combinations = n_features * (n_features - 1) // 2
    
    print("Computing pairwise mutual information and linkage disequilibrium...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pairwise_mi_and_ld_single)(i, j, X, y)
        for i, j in tqdm(itertools.combinations(range(n_features), 2), total=total_combinations, desc="Pairwise MI and LD")
    )
    
    mi_matrix = np.zeros((n_features, n_features))
    p_values = np.zeros((n_features, n_features))
    ld_matrix = np.zeros((n_features, n_features))
    
    for i, j, mi, p_value, ld in results:
        mi_matrix[i, j] = mi_matrix[j, i] = mi
        p_values[i, j] = p_values[j, i] = p_value
        ld_matrix[i, j] = ld_matrix[j, i] = ld
    
    print("Pairwise mutual information and linkage disequilibrium computation complete.")
    return mi_matrix, p_values, ld_matrix

def compute_pairwise_mi_and_ld_single(i, j, X, y):
    mi = compute_mutual_information_2d(X[:, i], X[:, j])
    
    # Compute p-value using permutation test
    n_permutations = 1000
    permutation_mis = [compute_mutual_information_2d(np.random.permutation(X[:, i]), X[:, j]) for _ in range(n_permutations)]
    p_value = np.mean(np.array(permutation_mis) >= mi)
    
    # Compute LD r-squared
    ld = rogers_huff_r(X[:, i], X[:, j])
    
    # Sanity check
    if not 0 <= ld <= 1:
        print(f"Warning: LD value {ld} for features {i} and {j} is out of bounds [0, 1]")
        ld = np.clip(ld, 0, 1)
    
    return i, j, mi, p_value, ld

# Compute pairwise MI and LD
print("Computing pairwise mutual information and linkage disequilibrium...")
mi_matrix, p_values, ld_matrix = compute_pairwise_mi_and_ld(X_encoded, y)

# Prepare pairwise results
pairwise_results = []
for i in range(mi_matrix.shape[0]):
    for j in range(i+1, mi_matrix.shape[1]):
        pairwise_results.append((X.columns[i], X.columns[j], mi_matrix[i, j], p_values[i, j], ld_matrix[i, j]))

pairwise_df = pd.DataFrame(pairwise_results, columns=['Feature1', 'Feature2', 'MI', 'p_value', 'LD_r_squared'])
pairwise_df['MI_percent'] = pairwise_df['MI'] / np.log(2) * 100 
pairwise_df = pairwise_df.sort_values('MI_percent', ascending=False)

# Save pairwise results
pairwise_df.to_csv(f'{output_dir}/pairwise_interactions_with_LD.csv', index=False)

print("Pairwise analysis with LD complete. Results saved to CSV file.")

# Group by Chromosome and Add Dividers
print("Creating heatmap with chromosome dividers...")
sorted_columns = sorted(X.columns, key=lambda x: int(x.split('_')[0].replace('C', '')))
sorted_indices = [list(X.columns).index(col) for col in sorted_columns]

# Update the MI heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(mi_matrix[np.ix_(sorted_indices, sorted_indices)], 
            xticklabels=sorted_columns, yticklabels=sorted_columns, 
            cmap='Greys', vmin=0, vmax=np.max(mi_matrix))

current_chrom = 0
for i, col in enumerate(sorted_columns):
    chrom = int(col.split('_')[0].replace('C', ''))
    if chrom != current_chrom:
        plt.axhline(y=i, color='white', linewidth=2)
        plt.axvline(x=i, color='white', linewidth=2)
        plt.axhline(y=i, color='red', linewidth=1, linestyle='--')
        plt.axvline(x=i, color='red', linewidth=1, linestyle='--')
        
        current_chrom = chrom

plt.title('Pairwise Mutual Information by Chromosome')
plt.tight_layout()
plt.savefig(f'{output_dir}/pairwise_mi_heatmap_by_chrom.png', dpi=300)
plt.close()

print("Pairwise analysis complete. Results saved to CSV file and heatmap saved as PNG file.")

# Create histogram of LD r-squared scores
plt.figure(figsize=(10, 6))
plt.hist(pairwise_df['LD_r_squared'], bins=50, edgecolor='black')
plt.axvline(x=0.4, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of Pairwise LD r-squared Scores')
plt.xlabel('LD r-squared')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/pairwise_ld_r_squared_histogram.png')
plt.close()

# Create histogram of LD r-squared scores with 0.2 intervals
plt.figure(figsize=(10, 6))
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
plt.hist(pairwise_df['LD_r_squared'], bins=bins, edgecolor='black')
plt.title('Histogram of Pairwise LD r-squared Scores (0.2 intervals)')
plt.xlabel('LD r-squared')
plt.ylabel('Frequency')
plt.xticks(bins)
for i in range(len(bins)-1):
    count = ((pairwise_df['LD_r_squared'] >= bins[i]) & (pairwise_df['LD_r_squared'] < bins[i+1])).sum()
    plt.text((bins[i] + bins[i+1])/2, plt.gca().get_ylim()[1]/2, f'n={count}', 
             horizontalalignment='center', verticalalignment='center')
plt.savefig(f'{output_dir}/pairwise_ld_r_squared_histogram_0.2_intervals.png')
plt.close()

print("Pairwise LD r-squared histograms saved as PNG files.")

# Create histogram of p-values
plt.figure(figsize=(10, 6))
plt.hist(pairwise_df['p_value'], bins=50, edgecolor='black')
plt.axvline(x=0.02, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of Pairwise p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/pairwise_p_value_histogram.png')
plt.close()

print("Pairwise p-value histogram saved as PNG file.")

# Create filtered DataFrame for pairwise interactions
filtered_df = pairwise_df[(pairwise_df['p_value'] < 0.02) & (pairwise_df['LD_r_squared'] <= 0.4)]

# Count filtered features for pairwise interactions
total_features_pairwise = len(set(pairwise_df['Feature1']).union(set(pairwise_df['Feature2'])))
filtered_features_pairwise = len(set(filtered_df['Feature1']).union(set(filtered_df['Feature2'])))
filtered_out_pairwise = total_features_pairwise - filtered_features_pairwise

# Save filtered pairwise results
filtered_df.to_csv(f'{output_dir}/filtered_pairwise_interactions.csv', index=False)

print("Filtered pairwise analysis complete. Results saved to CSV file.")

# Create heatmap for filtered results
filtered_mi_matrix = np.zeros((len(X.columns), len(X.columns)))
for _, row in filtered_df.iterrows():
    i = list(X.columns).index(row['Feature1'])
    j = list(X.columns).index(row['Feature2'])
    filtered_mi_matrix[i, j] = filtered_mi_matrix[j, i] = row['MI']

# Group by Chromosome and Add Dividers for filtered results
print("Creating heatmap with chromosome dividers for filtered results...")
sorted_columns = sorted(X.columns, key=lambda x: int(x.split('_')[0].replace('C', '')))
sorted_indices = [list(X.columns).index(col) for col in sorted_columns]

plt.figure(figsize=(20, 16))
sns.heatmap(filtered_mi_matrix[np.ix_(sorted_indices, sorted_indices)], 
            xticklabels=sorted_columns, yticklabels=sorted_columns, 
            cmap='Greys', vmin=0, vmax=np.max(filtered_mi_matrix[filtered_mi_matrix > 0]))

current_chrom = 0
for i, col in enumerate(sorted_columns):
    chrom = int(col.split('_')[0].replace('C', ''))
    if chrom != current_chrom:
        plt.axhline(y=i, color='white', linewidth=2)
        plt.axvline(x=i, color='white', linewidth=2)
        plt.axhline(y=i, color='red', linewidth=1, linestyle='--')
        plt.axvline(x=i, color='red', linewidth=1, linestyle='--')
        
        current_chrom = chrom

plt.title('Filtered Pairwise Mutual Information by Chromosome')
plt.tight_layout()
plt.savefig(f'{output_dir}/filtered_pairwise_mi_heatmap_by_chrom.png', dpi=300)
plt.close()

print("Filtered pairwise analysis complete. Results saved to CSV file and heatmap saved as PNG file.")

# Create heatmap for LD r-squared (full pairwise analysis)
print("Creating heatmap for LD r-squared...")
plt.figure(figsize=(20, 16))
ld_matrix = np.zeros((len(X.columns), len(X.columns)))
for _, row in pairwise_df.iterrows():
    i = list(X.columns).index(row['Feature1'])
    j = list(X.columns).index(row['Feature2'])
    ld_matrix[i, j] = ld_matrix[j, i] = row['LD_r_squared']

sns.heatmap(ld_matrix[np.ix_(sorted_indices, sorted_indices)], 
            xticklabels=sorted_columns, yticklabels=sorted_columns, 
            cmap='Greys', vmin=0, vmax=np.max(ld_matrix))

current_chrom = 0
for i, col in enumerate(sorted_columns):
    chrom = int(col.split('_')[0].replace('C', ''))
    if chrom != current_chrom:
        plt.axhline(y=i, color='white', linewidth=2)
        plt.axvline(x=i, color='white', linewidth=2)
        plt.axhline(y=i, color='red', linewidth=1, linestyle='--')
        plt.axvline(x=i, color='red', linewidth=1, linestyle='--')
        current_chrom = chrom

plt.title('Pairwise Linkage Disequilibrium (r-squared) by Chromosome')
plt.tight_layout()
plt.savefig(f'{output_dir}/pairwise_ld_heatmap_by_chrom.png', dpi=300)
plt.close()

print("LD r-squared heatmap saved as PNG file.")

# Create heatmap for LD r-squared (filtered pairwise analysis)
print("Creating heatmap for filtered LD r-squared...")
plt.figure(figsize=(20, 16))
filtered_ld_matrix = np.zeros((len(X.columns), len(X.columns)))
for _, row in filtered_df.iterrows():
    i = list(X.columns).index(row['Feature1'])
    j = list(X.columns).index(row['Feature2'])
    filtered_ld_matrix[i, j] = filtered_ld_matrix[j, i] = row['LD_r_squared']

sns.heatmap(filtered_ld_matrix[np.ix_(sorted_indices, sorted_indices)], 
            xticklabels=sorted_columns, yticklabels=sorted_columns, 
            cmap='Greys', vmin=0, vmax=np.max(filtered_ld_matrix[filtered_ld_matrix > 0])) 

current_chrom = 0
for i, col in enumerate(sorted_columns):
    chrom = int(col.split('_')[0].replace('C', ''))
    if chrom != current_chrom:
        plt.axhline(y=i, color='white', linewidth=2)
        plt.axvline(x=i, color='white', linewidth=2)
        plt.axhline(y=i, color='red', linewidth=1, linestyle='--')
        plt.axvline(x=i, color='red', linewidth=1, linestyle='--')
        current_chrom = chrom

plt.title('Filtered Pairwise Linkage Disequilibrium (r-squared) by Chromosome')
plt.tight_layout()
plt.savefig(f'{output_dir}/filtered_pairwise_ld_heatmap_by_chrom.png', dpi=300)
plt.close()

print("Filtered LD r-squared heatmap saved as PNG file.")

# Create histogram of filtered LD r-squared scores
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['LD_r_squared'], bins=50, edgecolor='black')
plt.axvline(x=0.4, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of Filtered Pairwise LD r-squared Scores')
plt.xlabel('LD r-squared')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/filtered_pairwise_ld_r_squared_histogram.png')
plt.close()

print("Filtered pairwise LD r-squared histogram saved as PNG file.")

# Create histogram of filtered p-values
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['p_value'], bins=50, edgecolor='black')
plt.axvline(x=0.02, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of Filtered Pairwise p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/filtered_pairwise_p_value_histogram.png')
plt.close()

print("Filtered pairwise p-value histogram saved as PNG file.")

def compute_interaction_information(X, y):
    H_X = entropy(np.histogramdd(X, bins=20)[0].flatten())
    H_y = entropy(np.histogram(y, bins=20)[0])
    y_reshaped = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
    H_Xy = entropy(np.histogramdd(np.column_stack((X, y_reshaped)), bins=20)[0].flatten())
    
    return H_X + H_y - H_Xy

def compute_threeway_r_squared(X, y):
    # Compute total variance
    total_var = np.var(y)
    
    # Fit a linear model with the three features
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    
    # Compute predicted values
    y_pred = X_with_const @ beta
    
    # Compute residual variance
    residual_var = np.var(y - y_pred)
    
    # Compute R-squared
    r_squared = 1 - (residual_var / total_var)
    
    return r_squared

def compute_threeway_mi_single(i, j, k, X, y):
    X_ijk = np.column_stack((X[:, i], X[:, j], X[:, k]))
    ii = compute_interaction_information(X_ijk, y)
    
    # Compute p-value using permutation test
    n_permutations = 100
    permutation_iis = [compute_interaction_information(X_ijk, np.random.permutation(y)) for _ in range(n_permutations)]
    p_value = np.mean(np.array(permutation_iis) >= ii)
    
    # Compute R-squared
    r_squared = compute_threeway_r_squared(X_ijk, y)
    
    return i, j, k, ii, p_value, r_squared

def compute_threeway_mi(X, y):
    n_features = X.shape[1]
    total_combinations = n_features * (n_features - 1) * (n_features - 2) // 6
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_threeway_mi_single)(i, j, k, X, y)
        for i, j, k in tqdm(itertools.combinations(range(n_features), 3), total=total_combinations, desc="3-way MI")
    )
    
    return results

# Compute 3-way mutual information
print("Computing 3-way mutual information...")
threeway_results = compute_threeway_mi(X_encoded, y)

threeway_df = pd.DataFrame(threeway_results, columns=['Feature1_idx', 'Feature2_idx', 'Feature3_idx', 'II', 'p_value', 'R_squared'])
threeway_df['Feature1'] = threeway_df['Feature1_idx'].map(dict(enumerate(X.columns)))
threeway_df['Feature2'] = threeway_df['Feature2_idx'].map(dict(enumerate(X.columns)))
threeway_df['Feature3'] = threeway_df['Feature3_idx'].map(dict(enumerate(X.columns)))
threeway_df['II_percent'] = threeway_df['II'] / np.log(2) * 100  # Convert to percentage
threeway_df = threeway_df.sort_values('II_percent', ascending=False)
threeway_df = threeway_df[['Feature1', 'Feature2', 'Feature3', 'II_percent', 'p_value', 'R_squared']]

threeway_df.to_csv(f'{output_dir}/3_way_interactions.csv', index=False)

print("3-way mutual information analysis complete. Results saved to CSV file.")

# Create filtered 3-way DataFrame
filtered_threeway_df = threeway_df[(threeway_df['p_value'] <= 0.001) & (threeway_df['R_squared'] <= 0.4)]

# Count filtered features for 3-way interactions
total_features_threeway = len(set(threeway_df['Feature1']).union(set(threeway_df['Feature2'])).union(set(threeway_df['Feature3'])))
filtered_features_threeway = len(set(filtered_threeway_df['Feature1']).union(set(filtered_threeway_df['Feature2'])).union(set(filtered_threeway_df['Feature3'])))
filtered_out_threeway = total_features_threeway - filtered_features_threeway

# Save filtered 3-way results
filtered_threeway_df.to_csv(f'{output_dir}/filtered_3_way_interactions.csv', index=False)

print("Filtered 3-way analysis complete. Results saved to CSV file.")

# Create a DataFrame with filtering information
filtering_info = pd.DataFrame({
    'Analysis': ['Pairwise', '3-way'],
    'Total Features': [total_features_pairwise, total_features_threeway],
    'Features After Filtering': [filtered_features_pairwise, filtered_features_threeway],
    'Features Filtered Out': [filtered_out_pairwise, filtered_out_threeway],
    'Percentage Filtered Out': [filtered_out_pairwise/total_features_pairwise*100, filtered_out_threeway/total_features_threeway*100]
})

# Save filtering information to CSV
filtering_info.to_csv(f'{output_dir}/filtering_summary.csv', index=False)

print("Filtering summary saved to CSV file.")

# Create histogram of 3-way R-squared scores
plt.figure(figsize=(10, 6))
plt.hist(threeway_df['R_squared'], bins=50, edgecolor='black')
plt.axvline(x=0.4, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of 3-way R-squared Scores')
plt.xlabel('R-squared')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/threeway_r_squared_histogram.png')
plt.close()

print("3-way R-squared histogram saved as PNG file.")

# Create histogram of 3-way p-values
plt.figure(figsize=(10, 6))
plt.hist(threeway_df['p_value'], bins=50, edgecolor='black')
plt.axvline(x=0.001, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of 3-way p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/threeway_p_value_histogram.png')
plt.close()

print("3-way p-value histogram saved as PNG file.")

# Create histogram of filtered 3-way p-values
plt.figure(figsize=(10, 6))
plt.hist(filtered_threeway_df['p_value'], bins=50, edgecolor='black')
plt.axvline(x=0.001, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of Filtered 3-way p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{output_dir}/filtered_threeway_p_value_histogram.png')
plt.close()

print("Filtered 3-way p-value histogram saved as PNG file.")

# Create combined histogram of pairwise and 3-way p-values
plt.figure(figsize=(12, 6))

# Plot pairwise p-values
plt.hist(pairwise_df['p_value'], bins=50, alpha=0.7, label='Pairwise', edgecolor='black')
plt.axvline(x=0.02, color='red', linestyle='--', label='Pairwise filter (0.02)')

# Plot 3-way p-values
plt.hist(threeway_df['p_value'], bins=50, alpha=0.7, label='3-way', edgecolor='black')
plt.axvline(x=0.001, color='blue', linestyle='--', label='3-way filter (0.001)')

plt.title('Histogram of Pairwise and 3-way p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.xscale('log')  # Use log scale for x-axis to better visualize small p-values
plt.savefig(f'{output_dir}/combined_pairwise_and_threeway_p_value_histogram.png')
plt.close()

print("Combined pairwise and 3-way p-value histogram saved as PNG file.")