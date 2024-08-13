import pandas as pd
import os
import numpy as np

INPUT_FILE = 'Part8a_output.csv'
OUTPUT_FILE = 'Part8b_output.csv'

def load_large_csv(file_path):
    print(f"Loading dataset...")
    try:
        # Verify the file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"The file {file_path} is empty.")
        
        # Print file details
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print("First few lines of the file:")
        with open(file_path, 'r') as f:
            for _ in range(5):
                print(f.readline().strip())
        
        # Use pandas to read the CSV file
        df = pd.read_csv(file_path,
                         encoding='utf-8-sig',  # Handle potential BOM
                         on_bad_lines='warn',   # Warn about problematic lines
                         low_memory=False)      # Disable low memory warnings
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
    
    return None

def check_dataframe(df):
    print("\nChecking dataframe...")
    
    # Check for missing values
    print("\nChecking for missing, null, or NaN values...")
    missing_values = df.isnull().sum()
    missing_percentages = 100 * df.isnull().sum() / len(df)
    
    missing_summary = pd.concat([missing_values, missing_percentages], axis=1, keys=['Total Missing', 'Percent Missing'])
    missing_summary = missing_summary[missing_summary['Total Missing'] > 0].sort_values('Total Missing', ascending=False)
    
    if missing_summary.empty:
        print("No missing, null, or NaN values found in the dataset.")
    else:
        print("Columns with missing, null, or NaN values:")
        print(missing_summary)
    
    print(f"\nTotal number of missing values: {missing_values.sum()}")
    print(f"Percentage of missing values in the entire dataset: {100 * missing_values.sum() / df.size:.2f}%")

    # Check for -1 values
    print("\nChecking for '-1' values...")
    minus_one_counts = (df == -1).sum()
    minus_one_percentages = 100 * (df == -1).sum() / len(df)

    minus_one_summary = pd.concat([minus_one_counts, minus_one_percentages], axis=1, keys=['Total -1', 'Percent -1'])
    minus_one_summary = minus_one_summary[minus_one_summary['Total -1'] > 0].sort_values('Total -1', ascending=False)

    if minus_one_summary.empty:
        print("No '-1' values found in the dataset.")
    else:
        print("Columns with '-1' values:")
        print(minus_one_summary)

    print(f"\nTotal number of '-1' values: {minus_one_counts.sum()}")
    print(f"Percentage of '-1' values in the entire dataset: {100 * minus_one_counts.sum() / df.size:.2f}%")

    # Check column titles
    print("\nColumn titles:")
    print(df.columns.tolist())
    print(f"Total number of columns: {len(df.columns)}")

    # Check indexes
    print("\nFirst 5 index values:")
    print(df.index[:5].tolist())
    print(f"Total number of rows: {len(df.index)}")

def replace_minus_one(df):
    print("\nReplacing -1 values with random choices of 0, 1, or 2...")

    # Create a mask for -1 values
    minus_one_mask = df == -1

    # Count the total number of -1 values
    total_minus_one = minus_one_mask.sum().sum()

    # Replace -1 values with random choices
    df = df.mask(minus_one_mask, np.random.choice([0, 1, 2], size=df.shape))

    print(f"Replaced {total_minus_one} '-1' values with random choices of 0, 1, or 2.")
    return df

def main():
    print("Starting main process...")
    try:
        df = load_large_csv(INPUT_FILE)
        if df is None:
            raise ValueError("Failed to load dataset")

        check_dataframe(df)

        # Replace -1 values and save the updated dataframe
        df = replace_minus_one(df)

        # Print the first 5 index values
        print("\nFirst 5 index values before dropping:")
        print(df.index[:5].tolist())

        # Drop the index column
        df = df.reset_index(drop=True)

        # Print the first 5 index values after dropping
        print("\nFirst 5 index values after dropping:")
        print(df.index[:5].tolist())

        # Save the updated dataframe
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nUpdated dataframe saved as {OUTPUT_FILE}")

        # Check the updated dataframe
        print("\nChecking updated dataframe:")
        check_dataframe(df)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Main process completed.")

if __name__ == "__main__":
    main()