import os
import sys
import shutil

def extract_files(src_directory):
    files_to_extract = ["s1_full_data_0.json", "s1_model_0.cif", "s1_summary_confidences_0.json"]
    dst_directory = os.path.join(src_directory, "af_files")
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    for root, dirs, files in os.walk(src_directory):
        for file in files:
            if any(file.endswith(f) for f in files_to_extract):
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_directory, file)
                if os.path.exists(dst_file_path) and os.path.samefile(src_file_path, dst_file_path):
                    continue
                shutil.copy2(src_file_path, dst_file_path) # Copy to the "af_files" directory

def compare_folders(af_folder, pdb_folder):
    missing_in_pdb = []
    missing_in_af = []

    af_cif_files = [f for f in os.listdir(af_folder) if f.endswith("_model_0.cif")]
    af_pdb_ids = set(f.split('_')[1].lower() for f in af_cif_files)
    pdb_cif_files = [f for f in os.listdir(pdb_folder) if f.endswith(".cif")]
    pdb_pdb_ids = set(f.replace('.cif', '') for f in pdb_cif_files)

    for af_id in af_pdb_ids:
        if af_id not in pdb_pdb_ids:
            missing_in_pdb.append(f"{af_id}.cif")
    for pdb_id in pdb_pdb_ids:
        if pdb_id not in af_pdb_ids:
            missing_in_af.append(f"fold_{pdb_id}_s1_model_0.cif")

    if missing_in_pdb:
        print("Files missing in 'pdb' folder:")
        for file in missing_in_pdb:
            print(file)
    else:
        print("All files from 'af' are present in 'pdb'.")

    if missing_in_af:
        print("\nFiles missing in 'af' folder:")
        for file in missing_in_af:
            print(file)
    else:
        print("All files from 'pdb' are present in 'af'.")

def find_subfolder(src_directory, subfolder_name):
    for root, dirs, files in os.walk(src_directory):
        if subfolder_name in dirs:
            return os.path.join(root, subfolder_name)
    return None

def list_pdb_files(pdb_folder):
    pdb_cif_files = [f for f in os.listdir(pdb_folder) if f.endswith(".cif")]
    pdb_files_without_extension = sorted([f.replace('.cif', '') for f in pdb_cif_files])
    count = 0
    print("Files found in 'pdb' folder:")
    for file in pdb_files_without_extension:
        print(file)
        count += 1
    print(f"Total: {count}")

def remove_from_pdb_folder(af_folder, pdb_folder):
    af_files = set()
    for root, _, files in os.walk(af_folder):
        for file in files:
            if file.endswith(".cif"):
                base_name = file.split("_")[1]  # Get base name like "fold_6fql_s1_model_0"
                af_files.add(base_name.lower())  # Extract "6fql" from "fold_6fql_s1_model_0.cif"

    pdb_files = set()
    for root, _, files in os.walk(pdb_folder):
        for file in files:
            if file.endswith(".cif"):
                base_name = file.split(".")[0]  # Remove .cif extension
                pdb_files.add(base_name.lower())

    files_to_move = pdb_files - af_files
    print("files_to_delete", files_to_move)

    # Move files that are not in af_folder to the af_unsupported folder
    unsupported_folder = os.path.join(pdb_folder, 'af_pdb_check')
    if not os.path.exists(unsupported_folder):
        os.makedirs(unsupported_folder)
    for file_name in files_to_move:
        file_path = os.path.join(pdb_folder, f"{file_name}.cif")
        if os.path.exists(file_path):
            new_path = os.path.join(unsupported_folder, f"{file_name}.cif")
            shutil.move(file_path, new_path)
            print(f"Moved: {file_path} to {new_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <protein_rna_folder>")
        sys.exit(1)

    src_directory = sys.argv[1]
    extract_files(src_directory)
    # af_folder = find_subfolder(src_directory, "af")
    # pdb_folder = find_subfolder(src_directory, "pdb")
    # print(af_folder, pdb_folder)
    # compare_folders(af_folder, pdb_folder)
    # remove_from_pdb_folder(af_folder, pdb_folder)
    # list_pdb_files(pdb_folder)