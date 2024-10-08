import subprocess
import time
import csv
from datetime import datetime
import os

def run_script(script_name):
    print(f"Starting {script_name}")
    start_time = time.time()
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return script_name, start_time_str, None, None

    end_time = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    duration = end_time - start_time
    duration_str = time.strftime('%H:%M:%S', time.gmtime(duration))
    print(f"{script_name} took {duration:.2f} seconds to complete")
    return script_name, start_time_str, end_time_str, duration_str

def main():
    scripts = ['FS_RF.py','FS_TuRF.py','FS_graph.py','FS_merge.py']
    
    # Check if all scripts exist
    missing_scripts = [script for script in scripts if not os.path.isfile(script)]
    if missing_scripts:
        print(f"Error: The following scripts are missing: {', '.join(missing_scripts)}")
        return

    results = []

    for script in scripts:
        result = run_script(script)
        results.append(result)
        if result[2] is None:  # Check if end_time_str is None, indicating an error
            print(f"Stopping execution due to error in {script}")
            break

    print("All scripts have been executed")

    # Write results to CSV
    with open('Master_script_times.csv', 'w', newline='') as csvfile:
        fieldnames = ['Script Name', 'Start Time', 'End Time', 'Duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({
                'Script Name': result[0],
                'Start Time': result[1],
                'End Time': result[2],
                'Duration': result[3]
            })

if __name__ == "__main__":
    main()
