import os
import subprocess
import shutil
import gemmi

# First use the first part, then the second uncommented part:
# https://gemmi.readthedocs.io/en/latest/program.html  gemmi convert [options] INPUT_FILE OUTPUT_FILE
# Firstly,  --rename-chain=OLD:NEW  Rename chain OLD to NEW (--rename-chain=:A adds
#                           missing chain IDs).
# Finally, copy-paste up to
# loop_
# _entity_poly.entity_id
def get_new_chain_id(index):
    """Generate new chain ID based on index, using A-Z first, then 0-9."""
    allowed_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")  # 36 unique IDs
    if index < len(allowed_chars):
        return allowed_chars[index]
    else:
        raise ValueError("Exceeded maximum number of unique chain IDs (A-Z, 0-9, a-z).")

def get_chain_ids_from_cif(file_path):
    """Extracts unique chain IDs from an mmCIF file using gemmi."""
    structure = gemmi.read_structure(file_path)
    return sorted(set(chain.name for model in structure for chain in model))

def rename_chains_with_gemmi(input_file, output_file):
    """Rename chains in an mmCIF file using `gemmi convert`."""
    chain_ids = get_chain_ids_from_cif(input_file)
    mapping = {old: get_new_chain_id(idx) for idx, old in enumerate(chain_ids)}

    # Construct gemmi convert command with multiple --rename-chain options
    rename_args = " ".join([f"--rename-chain={old}:{new}" for old, new in mapping.items()])

    command = f"gemmi convert {rename_args} {input_file} {output_file}"
    subprocess.run(command, shell=True, check=True)

    print(f"Renamed chains in {input_file} -> {output_file}: {mapping}")

def process_folder_rename(input_folder, output_folder):
    """Process all CIF files in a given folder."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".cif"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            try:
                rename_chains_with_gemmi(input_file, output_file)
            except ValueError as e:
                print(f"Error processing {filename}: {e}")
            except subprocess.CalledProcessError:
                print(f"Error running gemmi on {filename}")

def merge_file_contents(pdb_folder, pdb_renamed_folder):
    """Merge content from original and renamed files."""
    # Get all CIF files in both folders
    pdb_files = {f[:-4]: os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith(".cif")}
    pdb_renamed_files = {f[:-4]: os.path.join(pdb_renamed_folder, f) for f in os.listdir(pdb_renamed_folder) if
                         f.endswith(".cif")}

    # Process matching files
    for name in pdb_files:
        if name in pdb_renamed_files:
            with open(pdb_files[name], "r") as f:
                pdb_content = f.read()

            with open(pdb_renamed_files[name], "r") as f:
                pdb_renamed_content = f.read()

            # Find the entity_poly.entity_id line
            marker = "_entity_poly.entity_id"

            if marker in pdb_content and marker in pdb_renamed_content:
                # Extract content before the marker from the original pdb file
                prefix_pdb = pdb_content.split(marker)[0]
                # Extract content after the marker from the renamed pdb file
                suffix_renamed = pdb_renamed_content.split(marker, 1)[1]

                # Construct new content
                new_content = prefix_pdb + marker + suffix_renamed

                # Backup the original renamed file
                shutil.copy(pdb_renamed_files[name], pdb_renamed_files[name] + ".bak")

                # Write back the modified content
                with open(pdb_renamed_files[name], "w") as f:
                    f.write(new_content)

                print(f"Updated {name}.cif in {pdb_renamed_folder}")
            else:
                print(f"Skipping {name}.cif - marker not found in both files.")

if __name__ == "__main__":
    # Define folder paths
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "protein_rna")
    pdb_folder = os.path.join(data, "untitled")
    pdb_renamed_folder = os.path.join(data, "renamed")

    # Step 1: Rename chains using gemmi and save to renamed folder
    print("Step 1: Renaming chains...")
    print(f"Gemmi version: {gemmi.__version__}")
    process_folder_rename(pdb_folder, pdb_renamed_folder)

    # Step 2: Merge file contents in the renamed folder
    print("\nStep 2: Merging file contents...")
    merge_file_contents(pdb_folder, pdb_renamed_folder)

#
# import os
# import shutil
#
# # Define folder paths
# pdb_folder = "data/protein_rna/pdb_origin"
# pdb_renamed_folder = "data/protein_rna/pdb_rename"
#
# # Get all CIF files in both folders
# pdb_files = {f[:-4]: os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith(".cif")}
# pdb_renamed_files = {f[:-4]: os.path.join(pdb_renamed_folder, f) for f in os.listdir(pdb_renamed_folder) if
#                      f.endswith(".cif")}
#
# # Process matching files
# for name in pdb_files:
#     if name in pdb_renamed_files:
#         with open(pdb_files[name], "r") as f:
#             pdb_content = f.read()
#
#         with open(pdb_renamed_files[name], "r") as f:
#             pdb_renamed_content = f.read()
#
#         # Find the entity_poly.entity_id line
#         marker = "_entity_poly.entity_id"
#
#         if marker in pdb_content and marker in pdb_renamed_content:
#             # Extract content before the marker from the original pdb file
#             prefix_pdb = pdb_content.split(marker)[0]
#             # Extract content after the marker from the renamed pdb file
#             suffix_renamed = pdb_renamed_content.split(marker, 1)[1]
#
#             # Construct new content
#             new_content = prefix_pdb + marker + suffix_renamed
#
#             # Backup the original renamed file
#             shutil.copy(pdb_renamed_files[name], pdb_renamed_files[name] + ".bak")
#
#             # Write back the modified content
#             with open(pdb_renamed_files[name], "w") as f:
#                 f.write(new_content)
#
#             print(f"Updated {name}.cif in {pdb_renamed_folder}")
#         else:
#             print(f"Skipping {name}.cif - marker not found in both files.")
