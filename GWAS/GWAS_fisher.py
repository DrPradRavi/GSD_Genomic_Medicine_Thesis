import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import os
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

TARGET_COLUMN = 'Case1_Control0'
OUTPUT_DIR = 'GWAS_fishers'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GTF_FILE = 'GTF_file_condensed.csv'
genes_of_interest = [
    'COL1A1', 'COL1A2', 'CRTAP', 'P3H1', 'PPIB', 'FITM5', 'SERPINF1', 'SERPINH1', 
    'FKBP10', 'TMEM38B', 'BMP1', 'WNT1', 'CREB3L1', 'SPARC', 'TENT5A', 'MBTPS2', 
    'MESD', 'KDELR2', 'CCD134', 'SP7', 'IFITM5', 'P4HB', 'SEC24D'
]

def load_gtf_file():
    return pd.read_csv(GTF_FILE, dtype={'CHROM': str, 'POS_start': int, 'POS_end': int, 'gene_name': str})

def get_gene_coordinates(gtf_df):
    gene_coordinates = {}
    for gene in genes_of_interest:
        gene_info = gtf_df[gtf_df['gene_name'] == gene]
        if gene_info.empty:
            print(f"Warning: Could not find coordinates for {gene}")
            gene_coordinates[gene] = (None, None, None)
        else:
            gene_info = gene_info.iloc[0]
            gene_coordinates[gene] = (str(gene_info['CHROM']), gene_info['POS_start'], gene_info['POS_end'])
    return gene_coordinates

def map_feature_to_gene(feature, gene_coordinates):
    parts = feature.split('_')
    if len(parts) >= 2:
        chrom, pos = parts[0], int(parts[1])
        for gene, (gene_chrom, gene_start, gene_end) in gene_coordinates.items():
            if gene_chrom is not None and chrom == gene_chrom and gene_start <= pos <= gene_end:
                return gene
    return 'Not in target genes'

def load_large_csv(file_path):
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='warn', low_memory=False)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("First 5 values of the index:")
        print(df.iloc[:5, 0].tolist())
        return df
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def plot_p_value_histogram(p_values, output_dir, bonferroni_threshold):
    print("Plotting p-value histogram...")
    plt.figure(figsize=(12, 6), dpi=300)
    n, bins, patches = plt.hist(p_values, bins=100, edgecolor='black', linewidth=1, color=(239/255, 45/255, 81/255))
    
    top_3_indices = np.argsort(n)[-3:]
    for i in top_3_indices:
        plt.text((bins[i] + bins[i+1]) / 2, n[i], str(int(n[i])), ha='center', va='bottom')
    
    mean_p_value = np.mean(p_values)
    median_p_value = np.median(p_values)
    
    plt.axvline(mean_p_value, color='r', linestyle='--', label=f'Mean: {mean_p_value:.6f}')
    plt.axvline(median_p_value, color='b', linestyle='-', label=f'Median: {median_p_value:.6f}')
    
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Chi-Square p-values')
    plt.xlim(0, 1)
    plt.ylim(top=plt.ylim()[1] * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fisher_GWAS_p_value_histogram.png'))
    plt.close()
    print("p-value histogram plotted and saved.")

def plot_ordered_p_value_histogram(results_df, output_dir, bonferroni_threshold):
    print("Plotting ordered p-value histogram...")
    
    # Extract chromosome and position from feature names
    split_features = results_df['feature'].str.split('_', n=1, expand=True)
    results_df['CHROM'] = split_features[0]
    results_df['POS'] = split_features[1].str.split('_').str[0]  # Take the first part after CHROM
    
    # Convert POS to numeric, dropping any rows where conversion fails
    results_df['POS'] = pd.to_numeric(results_df['POS'], errors='coerce')
    results_df = results_df.dropna(subset=['CHROM', 'POS'])
    

    results_df = results_df[results_df['CHROM'] != 'Y']
    
    # Sort by chromosome and position
    results_df['CHROM'] = pd.Categorical(results_df['CHROM'], 
                                         categories=[str(i) for i in range(1, 23)],
                                         ordered=True)
    results_df = results_df.sort_values(['CHROM', 'POS'])
    
    plt.figure(figsize=(24, 12), dpi=300) 
    plt.plot(range(len(results_df)), -np.log10(results_df['p_value']), '.', color=(239/255, 45/255, 81/255), markersize=3)
    
    # Add chromosome boundaries
    chrom_boundaries = results_df.groupby('CHROM').size().cumsum()
    for boundary in chrom_boundaries[:-1]:
        plt.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.5)
    
    plt.xlabel('Chromosomal Position')
    plt.ylabel('-log10(p-value)')
    plt.title('Manhattan Plot of Chi-Square p-values')
    
    # Add chromosome labels
    chrom_centers = results_df.groupby('CHROM').size().cumsum() - results_df.groupby('CHROM').size() / 2
    plt.xticks(chrom_centers, chrom_centers.index, rotation=45)
    
    # Add significance thresholds
    plt.axhline(y=-np.log10(bonferroni_threshold), color='g', linestyle=':', label=f'Bonferroni threshold: {bonferroni_threshold}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fisher_GWAS_manhattan_plot.png'), dpi=300)
    plt.close()
    print("Manhattan plot saved.")

def process_chunk(chunk, y):
    results = []
    for col in chunk.columns:
        table = pd.crosstab(chunk[col], y)
        
        # Ensure the table is 2x2
        if table.shape != (2, 2):
            # If the table is not 2x2, we'll add a small epsilon to any zero counts
            epsilon = 1e-9
            if table.shape[0] < 2:
                table = table.reindex([0, 1], fill_value=epsilon)
            if table.shape[1] < 2:
                table = table.reindex(columns=[0, 1], fill_value=epsilon)
        
        try:
            _, p_value = fisher_exact(table)
        except ValueError:
            # If fisher_exact still fails, we'll assign a p-value of 1
            p_value = 1.0
        
        results.append((col, p_value))
    return results

def main():
    print("Starting Fisher's Exact Test Feature Selection process...")

    data = load_large_csv('Part8d_output.csv')
    if data is None:
        raise ValueError("Failed to load dataset")

    data.set_index(data.columns[0], inplace=True)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print(f"Dataset loaded. Shape: {X.shape}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Number of positive cases: {sum(y == 1)}")
    print(f"Number of negative cases: {sum(y == 0)}")

    print("\nPerforming Fisher's Exact test...")
    num_cores = max(1, multiprocessing.cpu_count() - 6)
    print(f"Using {num_cores} cores for processing")

    # Split X into chunks for parallel processing
    chunk_size = max(1, X.shape[1] // num_cores)
    X_chunks = [X.iloc[:, i:i+chunk_size] for i in range(0, X.shape[1], chunk_size)]

    # Process chunks in parallel
    results = Parallel(n_jobs=num_cores)(delayed(process_chunk)(chunk, y) for chunk in X_chunks)

    # Combine results
    all_results = [item for sublist in results for item in sublist]
    features, p_values = zip(*all_results)

    # Calculate Bonferroni corrected p-value threshold
    bonferroni_threshold = 0.05 / len(p_values)

    print(f"\nBonferroni corrected p-value threshold: {bonferroni_threshold}")

    results_df = pd.DataFrame({
        'feature': features,
        'p_value': p_values,
        'significant': np.array(p_values) < bonferroni_threshold
    }).sort_values('p_value')
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'Fisher_GWAS_all_features.csv'), index=False)

    # Plot ordered p-value histogram (Manhattan plot)
    plot_ordered_p_value_histogram(results_df, OUTPUT_DIR, bonferroni_threshold)

    # Save threshold values and feature counts
    threshold_info = pd.DataFrame({
        'threshold_type': ['Bonferroni'],
        'threshold_value': [bonferroni_threshold],
        'num_features': [sum(results_df['significant'])]
    })
    threshold_info.to_csv(os.path.join(OUTPUT_DIR, 'Fisher_GWAS_threshold_info.csv'), index=False)

    # Plot p-value histogram with Bonferroni threshold
    plot_p_value_histogram(p_values, OUTPUT_DIR, bonferroni_threshold)

    print("\nPerforming gene mapping...")
    gtf_df = load_gtf_file()
    gene_coordinates = get_gene_coordinates(gtf_df)
    
    results_df['gene'] = results_df['feature'].apply(lambda x: map_feature_to_gene(x, gene_coordinates))

    # Create gene summary
    gene_summary = pd.DataFrame(index=genes_of_interest + ['Not in target genes'])
    
    threshold_counts = results_df[results_df['significant']]['gene'].value_counts()
    gene_summary['significant_count'] = threshold_counts
    
    gene_summary = gene_summary.fillna(0).astype(int)
    gene_summary['all_features_count'] = results_df['gene'].value_counts()
    gene_summary = gene_summary.sort_values('all_features_count', ascending=False)
    gene_summary.to_csv(os.path.join(OUTPUT_DIR, 'Fisher_GWAS_gene_summary.csv'))

    # Plot gene distribution
    plt.figure(figsize=(24, 16), dpi=600)  # Increased DPI for higher resolution
    gene_summary_plot = gene_summary[gene_summary.index != 'Not in target genes']
    
    # Combine all features and significant features in one plot
    ax = gene_summary_plot[['all_features_count', 'significant_count']].plot(
        kind='bar', 
        color=[(0/255, 35/255, 102/255), (239/255, 45/255, 81/255)],
        width=0.8
    )
    plt.title('Distribution of Features Across Target Genes')
    plt.ylabel('Number of Features')
    plt.xlabel('Gene')
    plt.legend(['All Features', 'Significant Features'])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust y-axis to show more ticks
    max_features = int(gene_summary_plot['all_features_count'].max())
    plt.yticks(range(0, max_features + 1, max(1, max_features // 10)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fisher_GWAS_gene_distribution.png'), dpi=600, bbox_inches='tight')  # Increased DPI and added bbox_inches='tight'
    plt.close()

    print("\nGene mapping complete. Results saved in", OUTPUT_DIR)
    print(f"Total mapped features: {len(results_df)}")
    print(f"Features in target genes: {len(results_df[results_df['gene'] != 'Not in target genes'])}")
    print(f"Significant features in target genes: {gene_summary_plot['significant_count'].sum()}")
    print(f"Features not in target genes: {len(results_df[results_df['gene'] == 'Not in target genes'])}")

    # Print summary of features selected by the threshold
    print("\nFeatures selected by Bonferroni threshold:")
    print(f"Bonferroni threshold: {results_df['significant'].sum()}")
    print(f"Total unique features: {len(results_df)}")

    print("\nFeature selection complete. Results saved in", OUTPUT_DIR)
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features (Bonferroni): {sum(results_df['significant'])}")

if __name__ == '__main__':
    main()

