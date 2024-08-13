import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
import psutil

# Folder paths
CASE_INPUT_DIR = 'Part2_case_output'
CONTROL_INPUT_DIR = 'Part2_control_output'

def log_task(task_name, start_time, end_time):
    duration = end_time - start_time
    print("--------------------")
    print(f"Task: {task_name}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    print("--------------------")

def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        print(f"\rCPU Usage: {cpu_percent}% | Memory Usage: {memory_percent}%", end="", flush=True)
        time.sleep(1)

def process_file(file_path):
    print(f"Starting task: Processing {file_path}")
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Count the number of rows before removing duplicates
        initial_row_count = len(df)

        # Remove duplicate rows based on 'CHROM' and 'POS' columns
        df = df.drop_duplicates(subset=['CHROM', 'POS'])

        # Count the number of rows after removing duplicates
        final_row_count = len(df)

        # Calculate the number of removed rows
        removed_rows = initial_row_count - final_row_count

        # Replace any value in 'Genotype' column that isn't 0, 1, or 2 with -1
        df['Genotype'] = df['Genotype'].apply(lambda x: -1 if x not in [0, 1, 2, '0', '1', '2'] else x)

        # Convert data types
        df = df.astype({
            'CHROM': 'int8',
            'POS': 'int32',
            'ALT': 'category',
            'Genotype': 'category'
        })

        # Save the cleaned DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

        print(f"Processed {file_path}: Removed {removed_rows} rows")

        return removed_rows
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0

def main():
    # List all CSV files in the input directories
    csv_files = [os.path.join(CASE_INPUT_DIR, f) for f in os.listdir(CASE_INPUT_DIR) if f.endswith('.csv')]
    csv_files += [os.path.join(CONTROL_INPUT_DIR, f) for f in os.listdir(CONTROL_INPUT_DIR) if f.endswith('.csv')]

    total_removed_rows = 0
    total_files = len(csv_files)
    processed_files = 0

    # Start resource monitoring in a separate process
    resource_monitor = os.fork()
    if resource_monitor == 0:
        monitor_resources()
        return

    # Use multiprocessing Pool for parallel processing with 6 processes
    start_time = time.time()
    with Pool(processes=6) as pool:
        for result in pool.imap_unordered(process_file, csv_files):
            total_removed_rows += result
            processed_files += 1
            print(f"Progress: {processed_files}/{total_files} files processed")

    end_time = time.time()

    log_task("Process all CSV files", start_time, end_time)
    output_file = 'proof3isdone.csv'
    data = pd.DataFrame([[5]])
    data.to_csv(output_file, index=False, header=False)
    print(f"\nTotal rows removed: {total_removed_rows}")

    # Terminate the resource monitoring process
    os.kill(resource_monitor, 9)

if __name__ == "__main__":
    main()
