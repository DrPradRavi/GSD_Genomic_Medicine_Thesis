import os

def rename_files_in_folder(folder_path):
    for dir_name in os.listdir(folder_path):
        if dir_name.startswith(('Part10_', 'Part11_', 'Part12_', 'Part9_RF_', 'Part9_TuRF_')):
            dir_path = os.path.join(folder_path, dir_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        try:
                            name, ext = os.path.splitext(file_name)
                            if dir_name.startswith('Part9_RF_'):
                                new_file_name = f"{name}_RF{ext}"
                            elif dir_name.startswith('Part9_TuRF_'):
                                new_file_name = f"{name}_TuRF{ext}"
                            elif dir_name.endswith('_RF'):
                                new_file_name = f"{name}_RF{ext}"
                            else:
                                new_file_name = f"{name}_TURF{ext}"
                            new_file_path = os.path.join(dir_path, new_file_name)
                            os.rename(file_path, new_file_path)
                            print(f"Renamed {file_name} to {new_file_name}")
                        except Exception as e:
                            print(f"Error renaming {file_path}: {str(e)}")

# Run on the current directory
current_directory = os.getcwd()
rename_files_in_folder(current_directory)