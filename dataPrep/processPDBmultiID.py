import os
from gemmi import cif


# Path to the folder containing CIF files
import sys
input_folder = sys.argv[1] if len(sys.argv) > 1 else "data/protein_rna/pdb"   # folder of mmCIF files

for filename in os.listdir(input_folder):
    if filename.endswith(".cif"):
        file_path = os.path.join(input_folder, filename)

        doc = cif.read(file_path)

        # Modify the _atom_site columns
        for row in doc[0].find('_atom_site.', ['auth_seq_id', 'label_seq_id', 'pdbx_PDB_ins_code']):
            row[0] = row[1]  # Replace _atom_site.auth_seq_id with _atom_site.label_seq_id
            row[2] = '?'  # Set _atom_site.pdbx_PDB_ins_code to null ('?')

        doc.write_file(file_path)  # Overwrite the original file

        print(f"Processed: {file_path}")
