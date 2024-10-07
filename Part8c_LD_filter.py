import pandas as pd
import numpy as np
from scipy import stats
import os
import multiprocessing as mp

def calculate_r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def process_chromosome(chrom_data, window_size=2000, r2_threshold=0.6):
    features = chrom_data.columns[1:-1]
    to_keep = [features[0]]

    for i, feature in enumerate(features[1:], 1):
        pos_i = int(feature.split('_')[1])
        independent = True

        for kept_feature in to_keep:
            pos_j = int(kept_feature.split('_')[1])
            if abs(pos_i - pos_j) <= window_size:
                r2 = calculate_r2(chrom_data[feature], chrom_data[kept_feature])
                if r2 > r2_threshold:
                    independent = False
                    break

        if independent:
            to_keep.append(feature)

    return to_keep

def process_chunk(chunk):
    chrom, data = chunk
    kept_features = process_chromosome(data)
    return chrom, kept_features

if __name__ == '__main__':
    input_file = 'Part8b_output.csv'
    output_file = 'Part8c_output.csv'

    print("Reading input file...")
    df = pd.read_csv(input_file, index_col=0)

    # Option 1: Check column names
    print("Column names:")
    print(df.columns)

    # Option 2: Reset index
    print("Resetting index...")
    df = df.reset_index(drop=True)

    print("Grouping data by chromosome...")
    # Exclude the last column (assuming it's not a feature column)
    feature_columns = df.columns[:-1]
    grouped = df[feature_columns].groupby(lambda x: x.split('_')[0], axis=1)

    num_cores = os.cpu_count() - 6
    pool = mp.Pool(num_cores)

    print(f"Processing using {num_cores} cores...")
    results = []
    total_chroms = len(grouped)

    for i, (chrom, data) in enumerate(grouped, 1):
        results.append(pool.apply_async(process_chunk, args=((chrom, data),)))
        print(f"Processing chromosome {chrom} ({i}/{total_chroms})")

    pool.close()
    pool.join()

    print("Combining results...")
    kept_features = []
    for result in results:
        chrom, features = result.get()
        kept_features.extend(features)

    print("Filtering dataframe...")
    kept_features.append(df.columns[-1])
    df_pruned = df[kept_features]

    print("Saving output file...")
    df_pruned.to_csv(output_file)

    print(f"LD pruning complete. Output saved to {output_file}")
    print(f"Original features: {len(df.columns)}")
    print(f"Pruned features: {len(df_pruned.columns)}")
