import pandas as pd
import numpy as np
import os
import time

def process_file(input_file, output_file, step, case_sample_size=None, to_be_dropped_df=None):
    print(f"Starting Step {step} for file: {input_file}")
    start_time = time.time()
    
    df = pd.read_csv(input_file, index_col=0)
    predictive_values = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    
    total_columns = len(df.columns)
    progress_interval = max(1, total_columns // 100)  # 1% of columns
    
    if step == 1:
        abnormal_count = 0
        replaced_count = 0
        column_minus_one_counts = {}
        for i, col in enumerate(df.columns):
            if (i + 1) % progress_interval == 0:
                print(f"Step {step} Progress: {i+1}/{total_columns} columns processed ({((i+1)/total_columns)*100:.2f}%)")
            
            mask = ~df[col].isin([0, 1, 2])
            abnormal_count += mask.sum()
            if mask.any():
                df.loc[mask, col] = -1
                replaced_count += mask.sum()
            
            # Count -1 values in each column
            column_minus_one_counts[col] = (df[col] == -1).sum()
        
        # Create to_be_dropped_df if it's the case file
        if 'case' in input_file.lower():
            threshold = case_sample_size * 0.1
            to_be_dropped = [col for col, count in column_minus_one_counts.items() if count >= threshold]
            to_be_dropped_df = pd.DataFrame({'columns_to_drop': to_be_dropped})
            to_be_dropped_df.to_csv('Part8a_metric/columns_to_drop.csv', index=False)
        
        df = pd.concat([df, predictive_values], axis=1)
        df.to_csv(output_file)
        
        end_time = time.time()
        print(f"Finished Step {step} for file: {input_file}. Time taken: {end_time - start_time:.2f} seconds")
        return abnormal_count, replaced_count, to_be_dropped_df
    
    elif step == 2:
        # Drop columns listed in to_be_dropped_df
        if to_be_dropped_df is not None:
            df = df.drop(columns=to_be_dropped_df['columns_to_drop'])
        
        minus_one_count = 0
        replaced_count = 0
        replacement_methods = {'one_mode': 0, 'two_modes': 0, 'no_mode': 0}
        column_replacement_counts = {}
        
        for i, col in enumerate(df.columns):
            if (i + 1) % progress_interval == 0:
                print(f"Step {step} Progress: {i+1}/{total_columns} columns processed ({((i+1)/total_columns)*100:.2f}%)")
            
            mask = df[col] == -1
            minus_one_count += mask.sum()
            if mask.any():
                valid_values = df.loc[~mask, col]
                modes = valid_values.mode()
                if len(modes) == 1:
                    replacement = modes[0]
                    replacement_methods['one_mode'] += mask.sum()
                elif len(modes) == 2:
                    replacement = np.random.choice(modes, size=mask.sum())
                    replacement_methods['two_modes'] += mask.sum()
                else:
                    replacement = np.random.choice([0, 1, 2], size=mask.sum())
                    replacement_methods['no_mode'] += mask.sum()
                df.loc[mask, col] = replacement
                replaced_count += mask.sum()
                column_replacement_counts[col] = mask.sum()
        
        df = pd.concat([df, predictive_values], axis=1)
        df.to_csv(output_file)
        
        end_time = time.time()
        print(f"Finished Step {step} for file: {input_file}. Time taken: {end_time - start_time:.2f} seconds")
        return minus_one_count, replaced_count, replacement_methods, column_replacement_counts

# Create necessary folders
os.makedirs('Part8a', exist_ok=True)
os.makedirs('Part8a_metric', exist_ok=True)

# Read case sample size
case_sample_size = pd.read_csv('case_sample_size.csv', header=None).iloc[0, 0]

print("Starting Step 1")
# Step 1
control_abnormal, control_replaced, _ = process_file('Part8_intermediate_control/Part8_control.csv', 'Part8a/Part8a_intermediate_control.csv', 1, case_sample_size)
case_abnormal, case_replaced, to_be_dropped_df = process_file('Part8_intermediate_case/Part8_case.csv', 'Part8a/Part8a_intermediate_case.csv', 1, case_sample_size)

pd.DataFrame({
    'File': ['Control', 'Case'],
    'Abnormal Values': [control_abnormal, case_abnormal],
    'Replaced with -1': [control_replaced, case_replaced]
}).to_csv('Part8a_metric/step1_metrics.csv', index=False)
print("Finished Step 1")

print("Starting Step 2")
# Step 2
control_minus_one, control_replaced, control_methods, control_column_counts = process_file('Part8a/Part8a_intermediate_control.csv', 'Part8a/Part8a_intermediate2_control.csv', 2, to_be_dropped_df=to_be_dropped_df)
case_minus_one, case_replaced, case_methods, case_column_counts = process_file('Part8a/Part8a_intermediate_case.csv', 'Part8a/Part8a_intermediate2_case.csv', 2, to_be_dropped_df=to_be_dropped_df)

pd.DataFrame({
    'File': ['Control', 'Case'],
    '-1 Values': [control_minus_one, case_minus_one],
    'Replaced with 0, 1, or 2': [control_replaced, case_replaced]
}).to_csv('Part8a_metric/step2_metrics.csv', index=False)

# New CSV for replacement method counts
pd.DataFrame({
    'File': ['Control', 'Case'],
    'One Mode': [control_methods['one_mode'], case_methods['one_mode']],
    'Two Modes': [control_methods['two_modes'], case_methods['two_modes']],
    'No Mode': [control_methods['no_mode'], case_methods['no_mode']]
}).to_csv('Part8a_metric/replacement_method_counts.csv', index=False)

# New CSV for column replacement counts
combined_column_counts = {**control_column_counts, **case_column_counts}
pd.DataFrame.from_dict(combined_column_counts, orient='index', columns=['Replacement Count']).sort_values('Replacement Count', ascending=False).to_csv('Part8a_metric/column_replacement_counts.csv')

print("Finished Step 2")

print("Starting final concatenation and metrics")
# Concatenate final output
output_df = pd.concat([
    pd.read_csv('Part8a/Part8a_intermediate2_control.csv', index_col=0),
    pd.read_csv('Part8a/Part8a_intermediate2_case.csv', index_col=0)
])
output_df.to_csv('Part8a_output.csv')

# Count Case and Control indexes
index_counts = output_df.index.str.split('_').str[0].value_counts()

pd.DataFrame({
    'Type': index_counts.index,
    'Count': index_counts.values
}).to_csv('Part8a_metric/index_counts.csv', index=False)

# Count columns with -1 values
columns_with_minus_one = (output_df == -1).any().sum()

pd.DataFrame({
    'Metric': ['Columns with -1 values'],
    'Count': [columns_with_minus_one]
}).to_csv('Part8a_metric/columns_with_minus_one.csv', index=False)
print("Finished final concatenation and metrics")

print("Script completed successfully")
