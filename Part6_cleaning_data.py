import pandas as pd
import logging
import multiprocessing as mp
import numpy as np
import random
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, message="'DataFrame.swapaxes' is deprecated")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def replace_genotype(chunk):
    mask = chunk['Genotype'] == -1
    replaced_count = {
        'total_-1': sum(mask),
        'matches_found': 0,
        'single_highest_freq': 0,
        'multiple_highest_freq': 0,
        'random_present': 0,
        'random_no_match': 0
    }
    no_match_rows = []
    for idx, row in chunk[mask].iterrows():
        match = chunk[(chunk['CHROM'] == row['CHROM']) & 
                      (chunk['POS'] == row['POS']) & 
                      (chunk['ALT'] == row['ALT']) & 
                      (chunk['Genotype'].isin([0, 1, 2]))]
        if not match.empty:
            replaced_count['matches_found'] += 1
            max_freq = match['Freq'].max()
            max_freq_rows = match[match['Freq'] == max_freq]
            if len(max_freq_rows) == 1:
                chunk.loc[idx, 'Genotype'] = max_freq_rows['Genotype'].iloc[0]
                replaced_count['single_highest_freq'] += 1
            else:
                chunk.loc[idx, 'Genotype'] = random.choice(max_freq_rows['Genotype'].tolist())
                replaced_count['multiple_highest_freq'] += 1
        else:
            no_match_rows.append(row)
            # Check for present values in the row
            present_values = [val for val in [0, 1, 2] if val == chunk.loc[idx, 'Genotype']]
            if present_values:
                chunk.loc[idx, 'Genotype'] = random.choice(present_values)
                replaced_count['random_present'] += 1
            else:
                chunk.loc[idx, 'Genotype'] = random.choice([0, 1, 2])
                replaced_count['random_no_match'] += 1
    return chunk, replaced_count, no_match_rows

def sum_and_drop_duplicates(df):
    return df.groupby(['CHROM', 'POS', 'ALT', 'Genotype'], as_index=False)['Freq'].sum()

def process_group(group):
    if len(group) > 1:
        return group
    return None

def remove_singletypegenes(df):
    logging.info("Removing single-type genes")
    initial_count = df.groupby(['CHROM', 'POS']).ngroups
    logging.info(f"Initial unique combinations: {initial_count}")

    grouped = df.groupby(['CHROM', 'POS'])
    groups = [group for _, group in grouped]

    num_cores = max(1, mp.cpu_count() - 6)
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_group, groups),
            total=len(groups),
            desc="Processing groups",
            unit="group"
        ))

    filtered_df = pd.concat([result for result in results if result is not None])
    final_count = filtered_df.groupby(['CHROM', 'POS']).ngroups
    logging.info(f"Final unique combinations: {final_count}")

    return filtered_df, initial_count, final_count

def main():
    logging.info("Step 1: Running initial processing on the input file")
    df = pd.read_csv('Part5_output.csv')
    df, initial_count_1, final_count_1 = remove_singletypegenes(df)

    logging.info("Step 2: Running main processing")
    required_columns = {'CHROM', 'POS', 'ALT', 'Genotype', 'Freq'}
    if not required_columns.issubset(df.columns):
        logging.error(f"Missing required columns: {required_columns - set(df.columns)}")
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    if df.empty:
        logging.error("The DataFrame is empty.")
        raise ValueError("The DataFrame is empty.")

    num_cores = max(1, mp.cpu_count() - 3)
    chunks = np.array_split(df, num_cores)

    logging.info("Replacing -1 in 'Genotype' column with matching values")
    total_replaced = {
        'total_-1': 0,
        'matches_found': 0,
        'single_highest_freq': 0,
        'multiple_highest_freq': 0,
        'random_present': 0,
        'random_no_match': 0
    }
    all_no_match_rows = []
    with mp.Pool(num_cores) as pool:
        results = pool.map(replace_genotype, chunks)
        processed_chunks, replaced_counts, no_match_rows = zip(*results)
        for count in replaced_counts:
            for key in total_replaced:
                total_replaced[key] += count[key]
        all_no_match_rows.extend([item for sublist in no_match_rows for item in sublist])

    df = pd.concat(processed_chunks)
    df = sum_and_drop_duplicates(df)

    # Read case_sample_size.csv
    try:
        with open('case_sample_size.csv', 'r') as f:
            case_sample_size = int(f.read().strip())
    except FileNotFoundError:
        logging.error("case_sample_size.csv not found. Please ensure the file exists.")
        raise
    except ValueError:
        logging.error("Invalid value in case_sample_size.csv. Please ensure it contains a valid integer.")
        raise

    # Filter rows without matches
    threshold = 0.05 * 2 * case_sample_size
    filtered_no_match_rows = [row for row in all_no_match_rows if row['Freq'] <= threshold]
    
    removed_count = len(all_no_match_rows) - len(filtered_no_match_rows)
    logging.info(f"Removed {removed_count} rows without matches due to high frequency")

    # Save replacement count information
    logging.info("Saving replacement count information")
    os.makedirs('Part6', exist_ok=True)
    count_df = pd.DataFrame([total_replaced])
    count_df.to_csv('Part6/replacement_counts.csv', index=False)

    # Save number of -1 replacements
    replacements_df = pd.DataFrame({
        'Total_-1': [total_replaced['total_-1']],
        'Replaced': [total_replaced['matches_found'] + total_replaced['random_present'] + total_replaced['random_no_match'] + removed_count],
        'Removed_high_freq': [removed_count]
    })
    replacements_df.to_csv('Part6/minus_one_replacements.csv', index=False)

    # Save filtered rows without matches
    no_match_df = pd.DataFrame(filtered_no_match_rows)
    no_match_df.to_csv('Part6/rows_without_matches.csv', index=False)

    logging.info(f"Total -1 values: {total_replaced['total_-1']}")
    logging.info(f"Matches found: {total_replaced['matches_found']}")
    logging.info(f"No matches found: {total_replaced['total_-1'] - total_replaced['matches_found']}")
    for key in ['single_highest_freq', 'multiple_highest_freq', 'random_present', 'random_no_match']:
        logging.info(f"  {key}: {total_replaced[key]}")
    logging.info(f"Removed due to high frequency: {removed_count}")

    logging.info("Step 3: Running final processing on the output")
    df, initial_count_2, final_count_2 = remove_singletypegenes(df)

    logging.info("Saving final output")
    df.to_csv('Part6_output.csv', index=False)

    # Save count information
    logging.info("Saving count information")
    os.makedirs('Part6_counts', exist_ok=True)
    count_df = pd.DataFrame({
        'Stage': ['Initial', 'After dropping singletons', 'After replacing missing values', 'After dropping singletons again'],
        'UniqueCount': [initial_count_1, final_count_1, initial_count_2, final_count_2]
    })
    count_df.to_csv('Part6_counts/unique_combination_counts.csv', index=False)
    logging.info("Count information saved")

    logging.info("All tasks completed successfully")

if __name__ == "__main__":
    main()