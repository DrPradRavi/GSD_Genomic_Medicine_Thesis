import pandas as pd
import numpy as np
import os
from collections import Counter

def create_default_columns_df(input_file):
    print("\nStarting creation of default columns dataframe...")
    try:
        df = pd.read_csv(input_file)
        df['C_POS'] = df['CHROM'].astype(str) + '_' + df['POS'].map(str)
        
        # Custom sorting function
        def sort_key(c_pos):
            chrom, pos = c_pos.split('_')
            # Convert chromosome to integer if possible, otherwise keep as string
            try:
                chrom = int(chrom)
            except ValueError:
                pass
            return (chrom, int(pos))
        
        unique_c_pos = sorted(df['C_POS'].unique(), key=sort_key)
        default_columns = pd.Series(['sample'] + unique_c_pos, name='sample')
        print(default_columns.head())
        print("Finished creating default columns series.")
        return default_columns
    except Exception as e:
        print(f"Error in create_default_columns_df: {e}")
        print("DataFrame info:")
        print(df.info())
        print("\nFirst few rows of CHROM and POS:")
        print(df[['CHROM', 'POS']].head())
        raise

def process_files(input_dir, output_dir, prefix, default_columns):
    if os.path.exists(output_dir) and any(f.startswith(f'{prefix}_') and f.endswith('.csv') for f in os.listdir(output_dir)):
        print(f"Processed {prefix} files already exist in {output_dir}. Skipping processing.")
        return
    
    print(f"\nStarting processing of {prefix} files...")
    os.makedirs(output_dir, exist_ok=True)
    skipped_files = 0

    for i, filename in enumerate(sorted(os.listdir(input_dir)), 1):
        if filename.endswith('.csv'):
            print(f"Processing file {i}: {filename}")
            input_df = pd.read_csv(os.path.join(input_dir, filename))
            if 'C_POS' not in input_df.columns or 'Genotype' not in input_df.columns:
                print(f"Error: Required columns not found in {filename}. Skipping file.")
                print(f"Available columns: {input_df.columns}")
                continue
            df_name = f'{prefix}_{i}'
            df = pd.Series(index=default_columns, dtype='object')
            df.iloc[0] = df_name  # Set the 'sample' column value
            df.iloc[1:] = '-1'  # Set all other values to '-1'
            
            for _, row in input_df.iterrows():
                c_pos = row['C_POS']
                if c_pos in df.index:
                    df[c_pos] = row['Genotype']
            
            df = df.fillna(-1)
            
            genotype_counts = Counter(df.values[1:])  # Exclude 'sample' column
            minus_one_percentage = (genotype_counts[-1] / sum(genotype_counts.values())) * 100
            
            if minus_one_percentage < 5:
                output_file = os.path.join(output_dir, f'{df_name}.csv')
                df.to_frame().T.to_csv(output_file, index=False)
                print(f"  Saved: {output_file}")
            else:
                skipped_files += 1
                print(f"  Skipped: {filename} (>5% missing data)")
    
    print(f"\nFinished processing {prefix} files.")
    print(f"Number of {prefix} files skipped: {skipped_files}")

def concatenate_files(output_dir, prefix):
    final_output_file = os.path.join(output_dir, f'Part8_intermediate_{prefix.lower()}.csv')
    if os.path.exists(final_output_file):
        print(f"Concatenated file {final_output_file} already exists. Skipping concatenation.")
        return final_output_file

    print(f"\nStarting concatenation of {prefix} files...")
    fragment_dir = os.path.join(output_dir, 'concat_fragments')
    os.makedirs(fragment_dir, exist_ok=True)
    
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv') and f.startswith(f'{prefix}_')])
    fragment_count = 0
    
    for i in range(0, len(all_files), 50):  # Changed from 100 to 50
        fragment_count += 1
        fragment_files = all_files[i:i+50]  # Changed from i:i+100 to i:i+50
        fragment_dfs = []
        
        for filename in fragment_files:
            file_path = os.path.join(output_dir, filename)
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path, index_col=0)
            fragment_dfs.append(df)
        
        fragment_df = pd.concat(fragment_dfs, axis=0).sort_index()
        fragment_file = os.path.join(fragment_dir, f'{prefix.lower()}_fragment{fragment_count}.csv')
        fragment_df.to_csv(fragment_file, index=True)
        print(f"Saved fragment file: {fragment_file}")
        
        # Clear memory
        del fragment_dfs, fragment_df
        
    print(f"Created {fragment_count} fragments.")
    
    # Concatenate fragments
    print("Concatenating fragments...")
    final_dfs = []
    for filename in sorted(os.listdir(fragment_dir)):
        if filename.endswith('.csv') and filename.startswith(f'{prefix.lower()}_fragment'):
            file_path = os.path.join(fragment_dir, filename)
            print(f"Reading fragment: {file_path}")
            df = pd.read_csv(file_path, index_col=0)
            final_dfs.append(df)
    
    final_df = pd.concat(final_dfs, axis=0).sort_index()
    final_df.to_csv(final_output_file, index=True)
    print(f"Saved final concatenated file: {final_output_file}")
    print(f"Finished concatenation of {prefix} files.")
    return final_output_file

def process_intermediate_file(input_file, prefix):
    output_file = os.path.join(os.path.dirname(input_file), f'Part8_{prefix.lower()}.csv')
    if os.path.exists(output_file):
        print(f"Processed intermediate file {output_file} already exists. Skipping processing.")
        return output_file

    print(f"\nStarting processing of intermediate {prefix} file...")
    df = pd.read_csv(input_file, index_col=0)
    for column in df.columns[1:]:
        values = df[column].values
        value_counts = Counter(values[values != -1])
        
        if len(value_counts) > 0:
            modes = [k for k, v in value_counts.items() if v == max(value_counts.values()) and k in [0, 1, 2]]
            
            if len(modes) == 1:
                df[column] = df[column].replace(-1, modes[0])
            elif len(modes) == 2:
                mask = df[column] == -1
                replacement = np.random.choice(modes, size=mask.sum())
                df.loc[mask, column] = replacement
            else:
                mask = df[column] == -1
                replacement = np.random.choice([0, 1, 2], size=mask.sum())
                df.loc[mask, column] = replacement

    df['Case1_Control0'] = 1 if prefix == 'Case' else 0
    df.to_csv(output_file, index=True)
    print(f"Saved processed file: {output_file}")
    print(f"Finished processing intermediate {prefix} file.")
    return output_file

def combine_files(control_file, case_file):
    output_file = 'Part8_output.csv'
    if os.path.exists(output_file):
        print(f"Combined file {output_file} already exists. Skipping combination.")
        return pd.read_csv(output_file, index_col=0)

    print("\nStarting combination of control and case files...")
    control_df = pd.read_csv(control_file, index_col=0)
    case_df = pd.read_csv(case_file, index_col=0)
    combined_df = pd.concat([control_df, case_df], axis=0)
    combined_df.to_csv(output_file, index=True)
    print("Saved combined file: Part8_output.csv")
    print("Finished combination of control and case files.")
    return combined_df

def calculate_metrics(combined_df):
    metrics_dir = 'Part8_metrics'
    if os.path.exists(metrics_dir) and os.path.exists(os.path.join(metrics_dir, 'value_frequencies.csv')) and os.path.exists(os.path.join(metrics_dir, 'case_control_counts.csv')):
        print("Metrics files already exist. Skipping calculation.")
        return

    print("\nStarting calculation of metrics...")
    value_counts = combined_df.iloc[:, 1:-1].values.flatten().tolist()
    freq_df = pd.DataFrame(Counter(value_counts).items(), columns=['Value', 'Frequency'])
    freq_df = freq_df.sort_values('Value')

    os.makedirs(metrics_dir, exist_ok=True)
    freq_df.to_csv(os.path.join(metrics_dir, 'value_frequencies.csv'), index=False)
    print("Saved value frequencies: Part8_metrics/value_frequencies.csv")

    case_count = sum(1 for idx in combined_df.index if idx.startswith('Case_'))
    control_count = sum(1 for idx in combined_df.index if idx.startswith('Control_'))

    counts_df = pd.DataFrame({'Type': ['Case', 'Control'], 'Count': [case_count, control_count]})
    counts_df.to_csv(os.path.join(metrics_dir, 'case_control_counts.csv'), index=False)
    print("Saved case-control counts: Part8_metrics/case_control_counts.csv")

    print(f"\nCase count: {case_count}")
    print(f"Control count: {control_count}")
    print("Metrics saved in 'Part8_metrics' folder")
    print("Finished calculation of metrics.")

def main():
    print("Starting script execution...")
    
    print("\nStep 1: Creating default columns dataframe")
    default_columns_df = create_default_columns_df('Part6_output.csv')

    print("\nStep 2a: Processing case files")
    process_files('Part7_intermediate_case', 'Part8_intermediate_case', 'Case', default_columns_df)

    print("\nStep 2b: Processing control files")
    process_files('Part7_intermediate_control', 'Part8_intermediate_control', 'Control', default_columns_df)

    print("\nStep 3a: Concatenating case files")
    intermediate_case_file = concatenate_files('Part8_intermediate_case', 'Case')

    print("\nStep 3b: Concatenating control files")
    intermediate_control_file = concatenate_files('Part8_intermediate_control', 'Control')

    print("\nStep 4a: Processing intermediate case file")
    processed_case_file = process_intermediate_file(intermediate_case_file, 'Case')

    print("\nStep 4b: Processing intermediate control file")
    processed_control_file = process_intermediate_file(intermediate_control_file, 'Control')

    if os.path.exists(processed_case_file) and os.path.exists(processed_control_file):
        print("\nStep 5: Combining control and case files")
        combined_df = combine_files(processed_control_file, processed_case_file)

        print("\nStep 6: Calculating metrics")
        calculate_metrics(combined_df)
    else:
        print("\nSkipping Steps 5 and 6: Processed case or control file is missing.")

    print("\nScript execution completed.")

if __name__ == "__main__":
    main()