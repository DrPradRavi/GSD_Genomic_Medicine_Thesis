import os
import gzip
import csv
import time
import multiprocessing as mp
import pandas as pd
import random

# File and folder paths
CASE_INPUT_CSV = 'Part1_case_paths_output.csv'
CONTROL_INPUT_CSV = 'Part1_control_paths_output.csv'
CASE_OUTPUT_DIR = 'Part2_case_output'
CONTROL_OUTPUT_DIR = 'Part2_control_output'
CASE_SAMPLE_SIZE_CSV = 'case_sample_size.csv'

def log_task(task_name, start_time, end_time):
    duration = end_time - start_time
    print("--------------------")
    print(f"Task: {task_name}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Duration: {duration:.2f} seconds")
    print("--------------------")

def process_vcf(args):
    file_path, case_control_flag, output_file = args
    start_time = time.time()
    print(f"Starting task: Process VCF file {file_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    genotype_map = {
        '0/0': '0', '0|0': '0',
        '0/1': '1', '0|1': '1', '0/2': '1', '0|2': '1',
        '1/1': '2', '1|1': '2', '1/2': '2', '1|2': '2',
        '2/2': '2', '2|2': '2'
    }

    total_rows = 0
    processed_rows = 0

    try:
        with gzip.open(file_path, 'rt') as f, open(output_file, 'w', newline='') as out_f:
            csv_writer = csv.writer(out_f)
            csv_writer.writerow(["CHROM", "POS", "ALT", "Genotype", "Case|Control"])

            for line in f:
                if line.startswith('#'):
                    continue

                total_rows += 1
                fields = line.strip().split('\t')

                if fields[6] != 'PASS':
                    continue

                chrom = fields[0]
                pos = fields[1]
                alt = fields[4] if fields[4] != '.' else fields[3]
                genotype = fields[-1].split(':')[0]

                # Check if chrom is a valid chromosome
                if chrom.startswith('chr'):
                    chrom = chrom[3:]

                if not chrom.isdigit():
                    continue

                chrom_num = int(chrom)
                if chrom_num > 22:
                    continue

                genotype = genotype_map.get(genotype)

                # Additional check for '.' in any column
                if '.' in [chrom, pos, alt]:
                    continue

                csv_writer.writerow([chrom, pos, alt, genotype, case_control_flag])
                processed_rows += 1

                if total_rows % 50000000 == 0:
                    print(f"Processed {processed_rows:,} rows out of {total_rows:,} total rows for {file_path}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Finished processing {file_path}")
        print(f"Total rows: {total_rows:,}")
        print(f"Processed rows: {processed_rows:,}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Overall processing speed: {total_rows / total_time:.2f} rows/second")
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise

    log_task(f"Process VCF file {file_path}", start_time, end_time)
    return processed_rows

def process_files(csv_path, output_folder, case_control_flag, max_files=None, random_selection=False):
    df_paths = pd.read_csv(csv_path)

    if random_selection:
        df_paths = df_paths.sample(frac=1).reset_index(drop=True)

    if max_files:
        df_paths = df_paths.head(max_files)

    total_tasks = len(df_paths)
    start_time = time.time()
    print(f"Starting task: Process {total_tasks} files from {csv_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(output_folder, exist_ok=True)

    existing_csv_files = set(f for f in os.listdir(output_folder) if f.endswith('.csv'))

    tasks = []
    for _, row in df_paths.iterrows():
        file_path = row['file_path']
        output_file = os.path.join(output_folder, os.path.basename(file_path).replace('.vcf.gz', '.csv'))

        if os.path.basename(output_file) in existing_csv_files:
            print(f"Skipping {file_path} as it has already been processed.")
            continue

        tasks.append((file_path, case_control_flag, output_file))

    with mp.Pool(processes=4) as pool:
        results = pool.map(process_vcf, tasks)

    total_processed_rows = sum(results)
    processed_files = len(results)

    end_time = time.time()
    log_task(f"Process {processed_files} files from {csv_path}", start_time, end_time)
    return total_processed_rows, processed_files

def get_csv_files_count(folder):
    """Returns the count of .csv files in the given folder."""
    return len([f for f in os.listdir(folder) if f.endswith('.csv')])

def get_csv_files(folder):
    """Returns a list of .csv files in the given folder."""
    return [f for f in os.listdir(folder) if f.endswith('.csv')]

def delete_random_file(folder, files):
    """Deletes a random file from the given list of files in the specified folder."""
    file_to_delete = random.choice(files)
    os.remove(os.path.join(folder, file_to_delete))
    print(f"Deleted {file_to_delete} from {folder}")

def balance_csv_files(case_folder, control_folder):
    """Balances the number of .csv files between the case and control folders."""
    while True:
        case_files_count = get_csv_files_count(case_folder)
        control_files_count = get_csv_files_count(control_folder)

        print(f"Case CSV files: {case_files_count}")
        print(f"Control CSV files: {control_files_count}")

        if control_files_count <= case_files_count:
            print("Control folder now has the same or fewer files than the case folder. Stopping.")
            break

        control_files = get_csv_files(control_folder)
        delete_random_file(control_folder, control_files)

if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"Starting overall script execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Process all case files
    case_processed_rows, case_files_processed = process_files(CASE_INPUT_CSV, CASE_OUTPUT_DIR, 1)
    print(f"Processed {case_files_processed} case files")

    # Read the number of case samples from the case_sample_size.csv file
    with open(CASE_SAMPLE_SIZE_CSV, 'r') as f:
        case_sample_size = int(f.read().strip())

    # Process control files based on the case sample size
    control_processed_rows, control_files_processed = process_files(CONTROL_INPUT_CSV, CONTROL_OUTPUT_DIR, 0, max_files=case_sample_size, random_selection=True)
    print(f"Processed {control_files_processed} control files")

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    print("\nConversion Summary:")
    print(f"Total case files processed: {case_files_processed}")
    print(f"Total control files processed: {control_files_processed}")
    print(f"Total case rows processed: {case_processed_rows:,}")
    print(f"Total control rows processed: {control_processed_rows:,}")
    print(f"Overall time: {overall_duration:.2f} seconds")
    print(f"Overall processing speed: {(case_processed_rows + control_processed_rows) / overall_duration:.2f} rows/second")

    # Final check of CSV file counts (now in a loop)
    max_iterations = 2  # Add a maximum number of iterations to avoid infinite loop
    iteration = 0
    while iteration < max_iterations:
        final_case_csv_count = len([f for f in os.listdir(CASE_OUTPUT_DIR) if f.endswith('.csv')])
        final_control_csv_count = len([f for f in os.listdir(CONTROL_OUTPUT_DIR) if f.endswith('.csv')])
        print("\nFinal CSV file count:")
        print(f"Case CSV files: {final_case_csv_count}")
        print(f"Control CSV files: {final_control_csv_count}")

        if final_control_csv_count >= final_case_csv_count:
            print("Control CSV file count is greater than or equal to case CSV file count. Exiting.")
            break
        else:
            print("Case and control CSV file counts are not equal. Processing more control files.")
            additional_files_needed = final_case_csv_count - final_control_csv_count
            additional_rows, additional_files = process_files(CONTROL_INPUT_CSV, CONTROL_OUTPUT_DIR, 0, max_files=additional_files_needed, random_selection=True)
            control_processed_rows += additional_rows
            control_files_processed += additional_files
        iteration += 1

    if iteration == max_iterations:
        print("Reached maximum iterations. Exiting loop.")

    print("\nFinal Conversion Summary:")
    print(f"Total case files processed: {case_files_processed}")
    print(f"Total control files processed: {control_files_processed}")
    print(f"Total case rows processed: {case_processed_rows:,}")
    print(f"Total control rows processed: {control_processed_rows:,}")
    print(f"Overall time: {overall_duration:.2f} seconds")
    print(f"Overall processing speed: {(case_processed_rows + control_processed_rows) / overall_duration:.2f} rows/second")

    # Balance the number of CSV files between case and control folders
    balance_csv_files(CASE_OUTPUT_DIR, CONTROL_OUTPUT_DIR)