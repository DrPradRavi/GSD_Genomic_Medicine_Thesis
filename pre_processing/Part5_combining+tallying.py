import os
import pandas as pd
from joblib import Parallel, delayed
import gc
import time

def process_file(file_path, part4_keys, output_dir, chunk_size=10000):
    output_filename = os.path.basename(file_path).replace('.csv', '_95sapproved.csv')
    output_path = os.path.join(output_dir, output_filename)

    # Skip processing if the output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {file_path}, output file already exists.")
        return output_filename

    filtered_dfs = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        df_keys = set(zip(chunk['CHROM'], chunk['POS']))  # Fixed line
        filtered_keys = df_keys.intersection(part4_keys)
        filtered_chunk = chunk[chunk.apply(lambda row: (row['CHROM'], row['POS']) in filtered_keys, axis=1)]
        filtered_dfs.append(filtered_chunk)

    filtered_df = pd.concat(filtered_dfs)

    # Drop 'Case|Control' column
    filtered_df = filtered_df.drop(columns=['Case|Control'])

    # Group by (CHROM, POS, ALT, Genotype) and calculate 'Freq'
    grouped = filtered_df.groupby(['CHROM', 'POS', 'ALT', 'Genotype']).size().reset_index(name='Freq')

    # Drop duplicates and keep only the first occurrence
    filtered_df = grouped.drop_duplicates(subset=['CHROM', 'POS', 'ALT', 'Genotype'])

    filtered_df.to_csv(output_path, index=False)

    # Clear RAM
    del filtered_df
    del filtered_dfs
    del grouped
    gc.collect()

    return output_filename

def process_files(skip_processing=False, skip_intermediate=False):
    if not skip_processing:
        # Read the Part4 output.csv from the Part4_output folder
        part4_input_path = os.path.join('Part4_output', 'Part4_output.csv')
        part4_df = pd.read_csv(part4_input_path)
        part4_keys = set(zip(part4_df['CHROM'], part4_df['POS']))

        # Define input and output directories
        input_dirs = ['Part2_control_output', 'Part2_case_output']
        output_dir = 'Part5_output'
        os.makedirs(output_dir, exist_ok=True)

        # Calculate total number of files to process
        total_files = sum(len(files) for input_dir in input_dirs for _, _, files in os.walk(input_dir))
        processed_files = 0

        for input_dir in input_dirs:
            print(f"Starting task for directory: {input_dir}")
            file_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith('.csv')]

            # Process files in parallel
            results = Parallel(n_jobs=5)(delayed(process_file)(file_path, part4_keys, output_dir) for file_path in file_paths)

            for i, result in enumerate(results, 1):
                processed_files += 1
                print(f"{processed_files}/{total_files} files have been processed")

            print(f"Processed {processed_files}/{total_files} files in {input_dir}")
            print(f"Finished task for directory: {input_dir}")

    # Concatenate all files in Part5_output in batches of 50
    print("Starting concatenation of all files in Part5_output")
    output_dir = 'Part5_output'
    intermediate_dir = os.path.join(output_dir, 'intermediates')
    os.makedirs(intermediate_dir, exist_ok=True)
    all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_95sapproved.csv')]
    total_concat_files = len(all_files)
    batch_size = 50
    batch_number = 1

    if not skip_intermediate:
        for batch_start in range(0, total_concat_files, batch_size):
            combined_df = pd.DataFrame()
            batch_files = all_files[batch_start:batch_start + batch_size]
            for i, file in enumerate(batch_files, 1):
                start_time = time.time()
                df = pd.read_csv(file)
                combined_df = pd.concat([combined_df, df])
                end_time = time.time()
                print(f"{batch_start + i}/{total_concat_files} files have been successfully concatenated in {end_time - start_time:.2f} seconds")

            # Group by (CHROM, POS, ALT, Genotype) and sum 'Freq'
            grouped = combined_df.groupby(['CHROM', 'POS', 'ALT', 'Genotype'], as_index=False)['Freq'].sum()

            intermediate_filename = os.path.join(intermediate_dir, f'Part5_intermediate_part{batch_number}.csv')
            grouped.to_csv(intermediate_filename, index=False)
            print(f"Saved intermediate file: {intermediate_filename}")

            # Clear RAM
            del combined_df
            del grouped
            gc.collect()

            batch_number += 1

    # Concatenate all intermediate files
    print("Starting final concatenation of all intermediate files")
    intermediate_files = [os.path.join(intermediate_dir, f) for f in os.listdir(intermediate_dir) if f.endswith('.csv')]
    combined_df = pd.DataFrame()
    for i, file in enumerate(intermediate_files, 1):
        start_time = time.time()
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df])
        end_time = time.time()
        print(f"{i}/{len(intermediate_files)} intermediate files have been successfully concatenated in {end_time - start_time:.2f} seconds")

    # Group by (CHROM, POS, ALT, Genotype) and sum 'Freq'
    grouped = combined_df.groupby(['CHROM', 'POS', 'ALT', 'Genotype'], as_index=False)['Freq'].sum()

    final_output_path = 'Part5_output.csv'
    grouped.to_csv(final_output_path, index=False)
    print(f"Finished concatenation of all files into {final_output_path}")

if __name__ == "__main__":
    process_files(skip_processing=False, skip_intermediate=False)
