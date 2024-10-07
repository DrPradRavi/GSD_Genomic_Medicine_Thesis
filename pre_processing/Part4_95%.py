import os
import pandas as pd
import gc
import math

# Define input and output directories
input_dirs = ['Part2_control_output', 'Part2_case_output']
output_dir = 'Part4_output'
fragments_dir = os.path.join(output_dir, 'fragments')
intermediate_dirs = [os.path.join(output_dir, f'intermediate{i}') for i in range(1, 4)]

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fragments_dir, exist_ok=True)
for dir in intermediate_dirs:
    os.makedirs(dir, exist_ok=True)

# Count the number of files in the input directories
total_input_files = sum(
    len(files)
    for input_dir in input_dirs
    for _, _, files in os.walk(input_dir)
)

# Check if the number of files exceeds 400
if total_input_files > 400:
    raise ValueError("This script was designed to work with under 400 samples.")

# Calculate intermediate cutoff
if total_input_files > 200:
    five_percent = math.ceil(total_input_files * 0.05)
    intermediate_cutoff = 30 - five_percent
else:
    intermediate_cutoff = 0

print(f"Total input files: {total_input_files}")
print(f"Intermediate cutoff: {intermediate_cutoff}")

# Function to process each file
def process_file(file_path, i, total_files):
    try:
        df = pd.read_csv(file_path)
        df = df[['CHROM', 'POS']]
        
        # Ensure 'CHROM' values are strings for comparison
        df['CHROM'] = df['CHROM'].astype(str)
        
        # Split dataframe by CHROM values
        chrom_groups = df.groupby('CHROM')
        
        for chrom, group in chrom_groups:
            output_path = os.path.join(fragments_dir, f"CHROM{chrom}_{os.path.basename(file_path)}")
            if not os.path.exists(output_path):
                group.to_csv(output_path, index=False)
        
        # Clear RAM
        del df
        del chrom_groups
        gc.collect()  # Explicitly call garbage collector

        print(f"{i}/{total_files} completed in processing files")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Gather all CSV files
file_paths = [
    os.path.join(root, file)
    for input_dir in input_dirs
    for root, _, files in os.walk(input_dir)
    for file in files if file.endswith('.csv')
]

# Function to combine fragments into intermediate files
def combine_fragments(chrom, i, total_chroms, intermediate_dir, part_size=30):
    try:
        chrom_files = sorted([f for f in os.listdir(fragments_dir) if f.startswith(f'CHROM{chrom}_')])
        if not chrom_files:
            print(f"No fragments found for Chromosome {chrom}")
            return
        
        part_num = 1
        df_combined = pd.DataFrame()
        files_added = 0
        
        for file in chrom_files:
            fragment_df = pd.read_csv(os.path.join(fragments_dir, file)).assign(Freq=1)
            df_combined = pd.concat([df_combined, fragment_df], ignore_index=True)
            files_added += 1
            print(f"{files_added}/{part_size} fragments have been processed for Chromosome {chrom} block {part_num}")
            
            if files_added == part_size:
                df_combined = df_combined.groupby(['CHROM', 'POS'], as_index=False)['Freq'].sum()
                
                # Drop rows with Freq below intermediate_cutoff
                if intermediate_cutoff > 0:
                    df_combined = df_combined[df_combined['Freq'] >= intermediate_cutoff]
                
                intermediate_file = os.path.join(intermediate_dir, f'CHROM{chrom}_part{part_num}.csv')
                df_combined.to_csv(intermediate_file, index=False)
                
                # Clear RAM
                del df_combined
                gc.collect()
                
                print(f"Combined {files_added} fragments for Chromosome {chrom} {i} in {total_chroms}")
                part_num += 1
                df_combined = pd.DataFrame()
                files_added = 0
        
        # Process any remaining files
        if not df_combined.empty:
            df_combined = df_combined.groupby(['CHROM', 'POS'], as_index=False)['Freq'].sum()
            
            # Do not apply intermediate_cutoff to the last part
            intermediate_file = os.path.join(intermediate_dir, f'CHROM{chrom}_part{part_num}.csv')
            df_combined.to_csv(intermediate_file, index=False)
            
            # Clear RAM
            del df_combined
            gc.collect()
            
            print(f"Combined {files_added} fragments for Chromosome {chrom} {i} in {total_chroms}")
    except Exception as e:
        print(f"Error combining fragments for Chromosome {chrom}: {e}")

# Function to combine intermediate files
def combine_intermediate_files(chrom, current_intermediate_dir, next_intermediate_dir, part_size=5):
    try:
        chrom_files = sorted([f for f in os.listdir(current_intermediate_dir) if f.startswith(f'CHROM{chrom}_part')])
        if not chrom_files:
            print(f"No intermediate files found for Chromosome {chrom} in {current_intermediate_dir}")
            return
        
        part_num = 1
        for j in range(0, len(chrom_files), part_size):
            part_files = chrom_files[j:j + part_size]
            df_combined = pd.DataFrame()
            
            for k, file in enumerate(part_files, 1):
                fragment_df = pd.read_csv(os.path.join(current_intermediate_dir, file))
                df_combined = pd.concat([df_combined, fragment_df], ignore_index=True)
                print(f"{k}/{len(part_files)} intermediate files have been processed for Chromosome {chrom} block {part_num}")
            
            df_combined = df_combined.groupby(['CHROM', 'POS'], as_index=False)['Freq'].sum()
            intermediate_file = os.path.join(next_intermediate_dir, f'CHROM{chrom}_part{part_num}.csv')
            df_combined.to_csv(intermediate_file, index=False)
            
            # Clear RAM
            del df_combined
            gc.collect()
            
            print(f"Combined {len(part_files)} intermediate files into {intermediate_file}")
            part_num += 1
    except Exception as e:
        print(f"Error combining intermediate files for Chromosome {chrom}: {e}")

# Function to process intermediate files for each chromosome
def process_chromosome(chrom, i, total_chroms):
    try:
        output_file = os.path.join(output_dir, f'Part4_grouped_CHROM{chrom}.csv')
        if os.path.exists(output_file):
            print(f"Output file for Chromosome {chrom} already exists. Skipping.")
            return f"Finished processing CHROM{chrom}"
        
        intermediate_file = os.path.join(intermediate_dirs[-1], f'CHROM{chrom}_part1.csv')
        if not os.path.exists(intermediate_file):
            print(f"Intermediate file for Chromosome {chrom} does not exist")
            return f"Error: Intermediate file for CHROM{chrom} not found"
        
        df_combined = pd.read_csv(intermediate_file)
        
        # Group by (CHROM, POS) and sum 'Freq'
        df_combined = df_combined.groupby(['CHROM', 'POS'], as_index=False)['Freq'].sum()
        
        # Calculate 95% threshold
        total_chrom_files = len([f for f in os.listdir(fragments_dir) if f.startswith(f'CHROM{chrom}_')])
        threshold = math.ceil(0.95 * total_chrom_files)
        
        # Remove rows with Freq below the threshold
        df_combined = df_combined[df_combined['Freq'] >= threshold]
        
        # Drop the 'Freq' column
        df_combined = df_combined.drop(columns=['Freq'])
        
        df_combined.to_csv(output_file, index=False)
        
        # Clear RAM
        del df_combined
        gc.collect()
        
        print(f"{i}/{total_chroms} completed in cleaning intermediate files for Chromosome {chrom}")
        return f"Finished processing CHROM{chrom}"
    except Exception as e:
        return f"Error processing CHROM{chrom}: {e}"

if __name__ == '__main__':
    # Check if fragments already exist
    if not os.listdir(fragments_dir):
        print("Starting task: Processing files")
        total_files = len(file_paths)
        for i, file_path in enumerate(file_paths, 1):
            process_file(file_path, i, total_files)
        print("Finished task: Processing files")
    else:
        print("Fragments already exist. Skipping file processing step.")
    
    print("Starting task: Combining fragments into intermediate1 files")
    total_chroms = 22
    for i in range(1, total_chroms + 1):
        combine_fragments(i, i, total_chroms, intermediate_dirs[0], 30)
    print("Finished task: Combining fragments into intermediate1 files")
    
    print("Starting task: Combining intermediate1 files into intermediate2 files")
    for i in range(1, total_chroms + 1):
        combine_intermediate_files(i, intermediate_dirs[0], intermediate_dirs[1], 5)
    print("Finished task: Combining intermediate1 files into intermediate2 files")
    
    print("Starting task: Combining intermediate2 files into intermediate3 files")
    for i in range(1, total_chroms + 1):
        combine_intermediate_files(i, intermediate_dirs[1], intermediate_dirs[2], 5)
    print("Finished task: Combining intermediate2 files into intermediate3 files")
    
    print("Starting task: Processing intermediate3 files")
    results = []
    for i in range(1, total_chroms + 1):
        result = process_chromosome(i, i, total_chroms)
        results.append(result)
        print(result)
    print("Finished task: Processing intermediate3 files")

    print("Starting task: Combining all chromosome files")
    try:
        all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('Part4_grouped_CHROM')]
        if all_files:
            combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
            combined_df.to_csv(os.path.join(output_dir, 'Part4_output.csv'), index=False)
            print("All chromosome files have been combined into Part4_output.csv")
        else:
            print("No chromosome files found to combine")
    except Exception as e:
        print(f"Error combining chromosome files: {e}")
    print("Finished task: Combining all chromosome files")
