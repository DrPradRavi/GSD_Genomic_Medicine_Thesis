import numpy as np
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import entropy
from tqdm import tqdm
import networkx as nx

# Ensure output directory exists
output_dir = 'MI_epistasis_ordinal'
os.makedirs(output_dir, exist_ok=True)

# Get the number of available cores
available_cores = multiprocessing.cpu_count()

# Set the number of cores to use (6 less than max, but at least 1)
n_jobs = max(1, available_cores - 6)

print(f"Using {n_jobs} cores for parallel processing.")

# Load the data
print("Loading data...")
df = pd.read_csv('ML_Merge_Output/merged_features.csv', index_col=0)
print("Data loaded successfully.")

# Reverse one-hot encoding
def reverse_one_hot(df):
    # Identify columns that follow the CHROM_POS_Genotype format
    chrom_pos_cols = [col for col in df.columns if len(col.split('_')) >= 3]
    
    # Group columns by CHROM_POS
    grouped_cols = {}
    for col in chrom_pos_cols:
        chrom_pos = '_'.join(col.split('_')[:-1])
        if chrom_pos not in grouped_cols:
            grouped_cols[chrom_pos] = []
        grouped_cols[chrom_pos].append(col)
    
    # Create new dataframe with reversed encoding
    new_data = {}
    for chrom_pos, cols in grouped_cols.items():
        if len(cols) == 3:  # Ensure we have all three genotypes
            new_data[chrom_pos] = df[cols].idxmax(axis=1).map({f"{chrom_pos}_0": 0, f"{chrom_pos}_1": 1, f"{chrom_pos}_2": 2})
    
    reversed_df = pd.DataFrame(new_data)
    
    # Add non-CHROM_POS columns (like the target variable) to the new dataframe
    for col in df.columns:
        if col not in chrom_pos_cols:
            reversed_df[col] = df[col]
    
    return reversed_df

# Apply reverse one-hot encoding
df_reversed = reverse_one_hot(df)
print("One-hot encoding reversed successfully.")

# Separate features and target
X = df_reversed.iloc[:, :-1]
y = df_reversed.iloc[:, -1]

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

# No need for OrdinalEncoder as the data is already in the desired format
X_encoded = X.values

def compute_mutual_information_2d(X, y):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    H_X = entropy(np.histogram(X, bins=100)[0])
    H_y = entropy(np.histogram(y, bins=100)[0])
    
    H_Xy = entropy(np.histogram2d(X[:, 0], y, bins=100)[0].flatten())
    
    if X.shape[1] > 1:
        H_Xy += entropy(np.histogram2d(X[:, 1], y, bins=100)[0].flatten())
    
    return H_X + H_y - H_Xy

def compute_pairwise_mi(X, y):
    n_features = X.shape[1]
    total_combinations = n_features * (n_features - 1) // 2
    
    print("Computing pairwise mutual information...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pairwise_mi_single)(i, j, X, y)
        for i, j in tqdm(itertools.combinations(range(n_features), 2), total=total_combinations, desc="Pairwise MI")
    )
    
    mi_matrix = np.zeros((n_features, n_features))
    p_values = np.zeros((n_features, n_features))
    
    for i, j, mi, p_value in results:
        mi_matrix[i, j] = mi_matrix[j, i] = mi
        p_values[i, j] = p_values[j, i] = p_value
    
    print("Pairwise mutual information computation complete.")
    return mi_matrix, p_values

def compute_pairwise_mi_single(i, j, X, y):
    mi = compute_mutual_information_2d(X[:, i], X[:, j])
    
    # Compute p-value using permutation test
    n_permutations = 1000
    permutation_mis = [compute_mutual_information_2d(np.random.permutation(X[:, i]), X[:, j]) for _ in range(n_permutations)]
    p_value = np.mean(np.array(permutation_mis) >= mi)
    
    return i, j, mi, p_value

def compute_main_effect(X, y):
    return compute_mutual_information_2d(X, y)

def compute_pairwise_ig(i, j, X, y):
    I_AB_D = compute_interaction_information(np.column_stack((X[:, i], X[:, j])), y)
    I_A_D = compute_main_effect(X[:, i], y)
    I_B_D = compute_main_effect(X[:, j], y)
    return I_AB_D - I_A_D - I_B_D

def compute_ternary_mi_single(i, j, k, X, y):
    X_ijk = np.column_stack((X[:, i], X[:, j], X[:, k]))
    ii = compute_interaction_information(X_ijk, y)
    
    # Compute p-value using permutation test
    n_permutations = 100
    permutation_iis = [compute_interaction_information(X_ijk, np.random.permutation(y)) for _ in range(n_permutations)]
    p_value = np.mean(np.array(permutation_iis) >= ii)
    
    return i, j, k, ii, p_value

def compute_ternary_mi(X, y):
    n_features = X.shape[1]
    total_combinations = n_features * (n_features - 1) * (n_features - 2) // 6
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_ternary_mi_single)(i, j, k, X, y)
        for i, j, k in tqdm(itertools.combinations(range(n_features), 3), total=total_combinations, desc="Ternary MI")
    )
    
    return results

def compute_interaction_information(X, y):
    H_X = entropy(np.histogramdd(X, bins=100)[0].flatten())
    H_y = entropy(np.histogram(y, bins=100)[0])
    y_reshaped = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
    H_Xy = entropy(np.histogramdd(np.column_stack((X, y_reshaped)), bins=100)[0].flatten())
    
    return H_X + H_y - H_Xy

# Compute pairwise MI
print("Computing pairwise mutual information...")
mi_matrix, p_values = compute_pairwise_mi(X_encoded, y)

# Prepare pairwise results
pairwise_results = []
for i in range(mi_matrix.shape[0]):
    for j in range(i+1, mi_matrix.shape[1]):
        pairwise_results.append((X.columns[i], X.columns[j], mi_matrix[i, j], p_values[i, j]))

pairwise_df = pd.DataFrame(pairwise_results, columns=['Feature1', 'Feature2', 'MI', 'p_value'])
pairwise_df['MI_percent'] = pairwise_df['MI'] / np.log(2) * 100 
pairwise_df = pairwise_df[['Feature1', 'Feature2', 'MI_percent', 'p_value']]
pairwise_df = pairwise_df.sort_values('MI_percent', ascending=False)

# Save pairwise results
pairwise_df.to_csv(f'{output_dir}/pairwise_interactions.csv', index=False)

print("Pairwise analysis complete. Results saved to CSV file.")

# Create filtered DataFrame for pairwise interactions
filtered_df = pairwise_df[pairwise_df['p_value'] < 0.02]

# Save filtered pairwise results
filtered_df.to_csv(f'{output_dir}/filtered_pairwise_interactions.csv', index=False)

print("Filtered pairwise analysis complete. Results saved to CSV file.")

# Create histogram of pairwise p-values before filtering
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

# Create histogram of pairwise MI percentage scores
plt.figure(figsize=(10, 6))
plt.hist(pairwise_df['MI_percent'], bins=100, edgecolor='black')
plt.title('Histogram of Pairwise Mutual Information Percentage Scores')
plt.xlabel('MI Score (%)')
plt.ylabel('Frequency')
plt.savefig(f'{output_dir}/pairwise_mi_percent_score_histogram.png')
plt.close()

print("Pairwise MI percentage score histogram saved as PNG file.")

# Compute ternary mutual information
print("Computing ternary mutual information...")
ternary_results = compute_ternary_mi(X_encoded, y)

ternary_df = pd.DataFrame(ternary_results, columns=['Feature1_idx', 'Feature2_idx', 'Feature3_idx', 'II', 'p_value'])
ternary_df['Feature1'] = ternary_df['Feature1_idx'].map(dict(enumerate(X.columns)))
ternary_df['Feature2'] = ternary_df['Feature2_idx'].map(dict(enumerate(X.columns)))
ternary_df['Feature3'] = ternary_df['Feature3_idx'].map(dict(enumerate(X.columns)))
ternary_df['II_percent'] = ternary_df['II'] / np.log(2) * 100  # Convert to percentage
ternary_df = ternary_df[['Feature1', 'Feature2', 'Feature3', 'II_percent', 'p_value']]
ternary_df = ternary_df.sort_values('II_percent', ascending=False)

ternary_df.to_csv(f'{output_dir}/ternary_interactions.csv', index=False)

print("Ternary mutual information analysis complete. Results saved to CSV file.")

# Create filtered ternary DataFrame
filtered_ternary_df = ternary_df[ternary_df['p_value'] <= 0.0005]

# Save filtered ternary results
filtered_ternary_df.to_csv(f'{output_dir}/filtered_ternary_interactions.csv', index=False)

print("Filtered ternary analysis complete. Results saved to CSV file.")

# Create histogram of ternary p-values before filtering
plt.figure(figsize=(10, 6))
plt.hist(ternary_df['p_value'], bins=100, edgecolor='black')
plt.axvline(x=0.001, color='red', linestyle='--', label='Filter threshold')
plt.title('Histogram of Ternary p-values')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.xscale('log')  # Use log scale for x-axis to better visualize small p-values
plt.savefig(f'{output_dir}/ternary_p_value_histogram.png')
plt.close()

print("Ternary p-value histogram saved as PNG file.")

# Create histogram of ternary II percentage scores
plt.figure(figsize=(10, 6))
plt.hist(ternary_df['II_percent'], bins=100, edgecolor='black')
plt.title('Histogram of Ternary Interaction Information Percentage Scores')
plt.xlabel('II Score (%)')
plt.ylabel('Frequency')
plt.savefig(f'{output_dir}/ternary_ii_percent_score_histogram.png')
plt.close()

print("Ternary II percentage score histogram saved as PNG file.")

# Function to create and save network graph
def create_network_graph(df, output_file, title):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Feature1'], row['Feature2'])
        if 'Feature3' in row:
            G.add_edge(row['Feature1'], row['Feature3'])
            G.add_edge(row['Feature2'], row['Feature3'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=8, font_weight='bold')
    
    # Adjust labels to fit inside nodes
    labels = nx.get_node_attributes(G, 'label')
    adjusted_labels = {node: '\n'.join(label.split('_')) for node, label in labels.items()}
    nx.draw_networkx_labels(G, pos, adjusted_labels, font_size=8)

    plt.title(title)
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Network graph saved as {output_file}")

# Create and save pairwise network graph
create_network_graph(filtered_df, f'{output_dir}/pairwise_network_graph.png', 'Pairwise Interactions Network')

# Create and save ternary network graph
create_network_graph(filtered_ternary_df, f'{output_dir}/ternary_network_graph.png', 'Ternary Interactions Network')

print("Network graphs created and saved.")