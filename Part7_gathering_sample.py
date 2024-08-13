import os
import pandas as pd

# Define paths
dictionary_file = 'Part6_output.csv'
case_input_dir = 'Part2_case_output'
control_input_dir = 'Part2_control_output'
case_output_dir = 'Part7_intermediate_case'
control_output_dir = 'Part7_intermediate_control'

# Create output directories if they don't exist
os.makedirs(case_output_dir, exist_ok=True)
os.makedirs(control_output_dir, exist_ok=True)

# Read the dictionary file and filter columns
print("Reading dictionary file...")
try:
    reference_df = pd.read_csv(dictionary_file, usecols=['CHROM', 'POS'])
    print("Dictionary file read and filtered.")
except Exception as e:
    print(f"Error reading dictionary file: {e}")
    exit(1)

def process_directory(input_dir, output_dir):
    files = os.listdir(input_dir)
    total_files = len(files)
    for idx, file in enumerate(files):
        print(f"Processing file {idx + 1}/{total_files}: {file}")
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Check if the output file already exists
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping processing for {file}.")
            continue
        
        try:
            # Read the input file
            working_df = pd.read_csv(input_path)
            
            # Filter rows based on reference dataframe
            working_df = working_df.merge(reference_df, on=['CHROM', 'POS'])
            
            # Create C_POS column
            working_df['C_POS'] = working_df['CHROM'].astype(str) + '_' + working_df['POS'].astype(str)
            
            # Reorder columns
            working_df = working_df[['C_POS', 'Genotype']]
            
            # Drop duplicate rows
            working_df = working_df.drop_duplicates()
            
            # Save the processed file
            working_df.to_csv(output_path, index=False)
            
            print(f"Finished processing file {idx + 1}/{total_files}: {file}")
            print(f"{idx + 1}/{total_files} files have been processed")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
        finally:
            del working_df  # Free up memory

# Process case directory
print("Starting processing of Part2_case_output directory...")
process_directory(case_input_dir, case_output_dir)
print("Finished processing of Part2_case_output directory.")

# Process control directory
print("Starting processing of Part2_control_output directory...")
process_directory(control_input_dir, control_output_dir)
print("Finished processing of Part2_control_output directory.")
