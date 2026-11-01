import os
import sys
import json
import glob
import re
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from collections import defaultdict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import logging
from collections import Counter

logging.basicConfig(level=logging.ERROR)

def determine_sequence_type(sequence, cif_file):
    cleaned_sequence = re.sub(r'\([A-Z0-9]+\)', '', sequence)
    # min_length = 10 #TEM 60=X-ray
    # Check if the sequence has only allowed characters and is long enough
    if 'U' in cleaned_sequence:  # RNA check first
        return "rnaSequence"
    elif any(aa in 'RNDQEHILKMFPSWYV' for aa in cleaned_sequence):  # Protein check last
        return "proteinChain"
    else:
        pdb_info = MMCIF2Dict(cif_file)
        entity_poly_type = pdb_info.get('_entity_poly.type', [])
        entity_poly_seq = pdb_info.get('_entity_poly.pdbx_seq_one_letter_code_can', [])
        # print("entity_poly_type", entity_poly_type)
        # print("entity_poly_seq", entity_poly_seq)
        
        # Clean and compare with each sequence
        best_match_idx = -1
        best_match_score = -1
        
        for idx, ref_seq in enumerate(entity_poly_seq):
            # Remove newlines and clean modifications
            clean_ref_seq = re.sub(r'\([A-Z0-9]+\)', '', ref_seq.replace('\n', ''))
            
            # Skip if either sequence is empty
            if not cleaned_sequence or not clean_ref_seq:
                continue
                
            # Calculate similarity score (length of common subsequence)
            common_length = 0
            for i in range(min(len(cleaned_sequence), len(clean_ref_seq))):
                if cleaned_sequence[i] == clean_ref_seq[i]:
                    common_length += 1
            
            # Normalize score by sequence length
            max_len = max(len(cleaned_sequence), len(clean_ref_seq))
            if max_len > 0:  # Safety check
                score = common_length / max_len
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = idx
        
        print("entity_poly_type[best_match_idx] ", entity_poly_type[best_match_idx] if best_match_idx != -1 else "No match found")
        
        if best_match_idx != -1:
            if entity_poly_type[best_match_idx] == 'polypeptide(L)':
                return "proteinChain"
            elif entity_poly_type[best_match_idx] == 'polyribonucleotide':
                return "rnaSequence"
            elif entity_poly_type[best_match_idx] == 'polydeoxyribonucleotide':
                return "dnaSequence"
    
    return print(f"Invalid sequence in {cif_file}.")

# AA: 4 atoms in the backbone + residue
# NT: 12 atoms in the backbone + A: 10, G: 11, C: 8, U: 8, T: 9
necessary_atom_count = {
    'GLY': 4, 'ALA': 5, 'VAL': 7, 'LEU': 8, 'ILE': 8, 'CYS': 6, 'MET': 8,
    'PHE': 11, 'TRP': 14, 'PRO': 7, 'SER': 6, 'THR': 7, 'TYR': 12,
    'ASN': 8, 'GLN': 9, 'ASP': 8, 'GLU': 9, 'HIS': 10,
    'LYS': 9, 'ARG': 11, 'C': 20, 'A': 22, 'G': 23, 'U': 20, 'T': 21
}
nucleotides = {'C', 'A', 'G', 'U', 'T'}
modified_residues_single = {'6MZ', 'PSU', '5MC', 'OMC', '4OC', '5MU', 'OMU', 'UR3', 'A2M', 'MA6', '6MZ', '2MG', 'OMG', '7MG',
                     'RSQ', '5CM', 'C34', '5HC', '6OG', '6MA', '1CC', '8OG', '5FC', '3DR', 'SEP', 'TPO', 'PTR',
                     'NEP', 'HIP', 'ALY', 'MLY', 'M3L', 'MLZ', '2MR', 'AGM', 'MCS', 'HYP', 'HY3', 'LYZ', 'AHB',
                     'P1L', 'SNN', 'SNC', 'TRF', 'KCR', 'CIR', 'YHA'}

modified_residues_nucleotide = {
    '6MZ': 'A', 'PSU': 'U', '5MC': 'C', 'OMC': 'C', '4OC': 'C', '5MU': 'U',
    'OMU': 'U', 'UR3': 'U', 'A2M': 'A', 'MA6': 'A', '2MG': 'G', 'OMG': 'G',
    '7MG': 'G', 'RSQ': 'G', '5CM': 'C', 'C34': 'C', '5HC': 'C', '6OG': 'G',
    '6MA': 'A', '1CC': 'C', '8OG': 'G', '5FC': 'C', '3DR': 'T'
}

modified_residues_protein = {'SEP': 'S',
    'TPO': 'T', 'PTR': 'Y', 'NEP': 'H', 'HIP': 'H', 'ALY': 'K', 'MLY': 'K',
    'M3L': 'K', 'MLZ': 'K', '2MR': 'R', 'AGM': 'R', 'MCS': 'C', 'HYP': 'P',
    'HY3': 'H', 'LYZ': 'K', 'AHB': 'A', 'P1L': 'P', 'SNN': 'S', 'SNC': 'C',
    'TRF': 'W', 'KCR': 'K', 'CIR': 'R', 'YHA': 'Y'
}


# Parse modified sequence in parentheses and return sequence with removed modification replaced by the unmodified moiety
def parse_modified_sequence(sequence, is_protein=False):
    modifications = []
    unmodified_sequence = []

    # Regex pattern to find modifications eg (6MZ)
    mod_pattern = re.compile(r'\(([A-Z0-9]{3})\)')
    mod_dict = modified_residues_protein if is_protein else modified_residues_nucleotide

    i = 0
    while i < len(sequence):
        if sequence[i] == '(':
            match = mod_pattern.match(sequence[i:])
            if match:
                moda = match.group(1)
                mod = f"CCD_{match.group(1)}"  # Extract modification (e.g., '6MZ') and add CCD_ for AF3
                # print('mod', mod)
                unmodified_residue = mod_dict.get(moda, '')  # Get unmodified residue
                if unmodified_residue:
                    unmodified_sequence.append(unmodified_residue)
                    if is_protein:
                        # Protein modification (ptmType and ptmPosition)
                        modifications.append({
                            "ptmType": mod,
                            "ptmPosition": len(unmodified_sequence)
                        })
                    else:
                        # Nucleotide modification (modificationType and basePosition)
                        modifications.append({
                            "modificationType": mod,
                            "basePosition": len(unmodified_sequence)
                        })

                # Skip modification in parentheses
                i += len(match.group(0))  # Move past the entire '(6MZ)'
                continue  # Skip appending
        #     else:
        #         # If no valid match is found, add the character
        #         unmodified_sequence.append(sequence[i])
        #         i += 1
        #         print("no valid match", sequence[i:])
        # else:
        #     # Normal character to the unmodified sequence
        #     unmodified_sequence.append(sequence[i])
        #     i += 1
        unmodified_sequence.append(sequence[i])
        i += 1

    return ''.join(unmodified_sequence), modifications

def is_resname_in_sequence(sequences, resname):
    # Regex pattern to match the residue name inside parentheses
    pattern = re.compile(rf'\({resname}\)')

    for sequence in sequences:
        if pattern.search(sequence):
            print(f"Residue '{resname}' found in sequence: {sequence}")
            return True
        # else:
        #     print(f"Residue '{resname}' not found in sequence: {sequence}")
    return False

def extract_ligands_ions(cif_file):
    pdb_info = MMCIF2Dict(cif_file)
    allowed_ligands = {
        "ADP", "ATP", "AMP", "GTP", "GDP", "FAD", "NAD", "NAP", "NDP",
        "HEM", "HEC", "PLM", "OLA", "MYR", "CIT", "CLA", "CHL", "BCL", "BCB"
    }
    allowed_ions = {"MG", "ZN", "CL", "CA", "NA", "MN", "K", "FE", "CU", "CO"}

    # Get atom site info
    comp_ids = pdb_info.get('_atom_site.label_comp_id', [])
    # If only one atom, comp_ids is a string, not a list
    if isinstance(comp_ids, str):
        comp_ids = [comp_ids]

    # Count occurrences of each allowed ligand/ion in atom_site
    comp_id_counts = Counter(comp_ids)

    ligand_ion_output = []

    for comp_id in allowed_ligands | allowed_ions:
        count = comp_id_counts.get(comp_id, 0)
        if count == 0:
            continue
        if comp_id in allowed_ligands:
            ligand_ion_output.append({
                "ligand": {
                    "ligand": f"CCD_{comp_id}",
                    "count": count
                }
            })
        elif comp_id in allowed_ions:
            ligand_ion_output.append({
                "ion": {
                    "ion": comp_id,
                    "count": count
                }
            })

    return ligand_ion_output

def parse_cif_sequence(cif_file):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)
    model = structure[0]
    sequences = defaultdict(list)
    pdb_info = MMCIF2Dict(cif_file)
    entity_poly_seq = pdb_info.get('_entity_poly.pdbx_seq_one_letter_code', [])
    entity_poly_type = pdb_info.get('_entity_poly.type', [])

    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            chain_id = chain.get_id()
            # Convert the three-letter code to the one-letter code if it is an amino acid
            if resname in modified_residues_single:
                sequences[chain_id].append(f"({resname})")
            elif resname in nucleotides:
                sequences[chain_id].append(resname)
            elif resname in necessary_atom_count and resname not in modified_residues_single and resname not in nucleotides:
                sequences[chain_id].append(seq1(resname))
            # Invalid ligands in chains:
            elif (resname == "ADP" or resname == "ATP" or resname == "AMP" or resname == "GDP" or resname == "GTP") and is_resname_in_sequence(entity_poly_seq, resname):
                print(f"ERROR: Invalid chain modification: {resname}, chain-id: {chain_id}.")
                return None
            # DNA
            elif resname in ['DA', 'DT', 'DG', 'DC'] and ('polyribonucleotide' not in entity_poly_type or ('polydeoxyribonucleotide' in entity_poly_type and 'polyribonucleotide' in entity_poly_type)):
                if resname == 'DA':
                    resname = 'A'
                elif resname == 'DT':
                    resname = 'T'
                elif resname == 'DG':
                    resname = 'G'
                elif resname == 'DC':
                    resname = 'C'
                sequences[chain_id].append(resname)
            # Modifications such as SRA, MSE, G46:
            elif (resname not in necessary_atom_count and resname not in modified_residues_single and resname not in nucleotides) and is_resname_in_sequence(entity_poly_seq, resname):
                print(f"ERROR: Invalid modification: {resname}, chain-id: {chain_id}.")
                return None

        for chain_id, residues in sequences.items():
            sequences[chain_id] = ''.join(residues)
            length = len(sequences[chain_id])
            if length < 4:
                print(f"Chain {chain} has a sequence shorter than 4 nucleotides/amino acids.")
                return None
    # for chain, sequence in sequences.items():
    #     length = len(sequence)
    #     if length < 4:
    #         print(f"Error {cif_file}, chain {chain}: sequence is too short: {len(sequence)}.")
    #         return None
    #     else:
    return sequences

def extract_reference_sequences(cif_file):
    pdb_info = MMCIF2Dict(cif_file)

    # entity_poly_type = pdb_info.get('_entity_poly.type', [])
    entity_poly_seq = pdb_info.get('_entity_poly.pdbx_seq_one_letter_code', [])
    entity_poly_seq = [seq.replace('\n', '') for seq in entity_poly_seq]
    strand_ids = pdb_info.get('_entity_poly.pdbx_strand_id', [])
    modified_nucleotides = pdb_info.get('_entity_poly.nstd_monomer', [])
    # print(entity_poly_type)
    print(entity_poly_seq)
    print(strand_ids)
    print(modified_nucleotides)

    reference_sequences = {}
    for chain_id, sequence in zip(strand_ids, entity_poly_seq):
        reference_sequences[chain_id] = sequence.replace('\n', '').strip()

    print("reference_sequences", reference_sequences)

    return reference_sequences

def save_combined_json(sequences, output_file):
    combined_json = []
    for base_name, seqs in sequences.items():
        job = {
            "name": f"{base_name}_ion_s1",
            "modelSeeds": [1],
            "sequences": seqs
        }
        combined_json.append(job)

    # with open(output_file, 'w') as json_file:
    #     json.dump(combined_json, json_file, indent=4)
    # Split combined_json into batches of 20 or 100 jobs
    batch_size = 30
    total_jobs = len(combined_json)
    batch_count = (total_jobs + batch_size - 1) // batch_size  # Calculate how many batches we need

    for batch_num in range(batch_count):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_jobs)
        batch_data = combined_json[start_idx:end_idx]
        # for idx, job in enumerate(batch_data):
        #     # Update "name" to have correct suffix, e.g., "_s1", "_s2", etc.
        #     job["name"] = f"{job['name'][:-2]}_s{batch_num + 1}"
        output_file_batch = output_file.replace('.json', f'_{batch_num + 1}.json')

        with open(output_file_batch, 'w') as json_file:
            json.dump(batch_data, json_file, indent=4)

        print(f"Saved {len(batch_data)} jobs to {output_file_batch}")


def process_cif_folder(folder_path, output_file):
    all_sequences = {}

    for cif_file in glob.glob(os.path.join(folder_path, "*.cif")):
        base_name = os.path.basename(cif_file).split('.')[0]  # Get the base name of the file (without extension)
        parsed_sequences = parse_cif_sequence(cif_file)  # Get the sequences from CIF file
        try:
            non_poly_info = extract_ligands_ions(cif_file)  # Get the ions and ligands
        except ValueError as e:
            print(f"Error in file {cif_file}: {e}")
            continue
        # for chain_id, sequence in parsed_sequences.items():
        #     print(f"Chain {chain_id}: {sequence}")

        if parsed_sequences is None:
            print(f"Discarding job for '{base_name}' due to invalid modifications.")
            continue

        # print(non_poly_info)
        if non_poly_info == []:  ##### Remove if also samples without ions or ligands should be allowed! #####
            print(f"Discarding job for '{base_name}' due to missing ions and ligands.")
            continue

        for chain_id, sequence in parsed_sequences.items():
            # Identify modifications and get the unmodified version
            is_protein = determine_sequence_type(sequence, cif_file) == "proteinChain"
            unmodified_sequence, modifications = parse_modified_sequence(sequence, is_protein)

            sequence_type = determine_sequence_type(unmodified_sequence, cif_file)

            if sequence_type:
                sequence_data = {
                    sequence_type: {
                        "sequence": unmodified_sequence,
                        "modifications": modifications,
                        "count": 1
                    }
                }
                if base_name not in all_sequences:
                    all_sequences[base_name] = []
                all_sequences[base_name].append(sequence_data)

        if non_poly_info:
            if base_name not in all_sequences:
                all_sequences[base_name] = []
            all_sequences[base_name].extend(non_poly_info)

    save_combined_json(all_sequences, output_file)


def split_json_to_batches(combined_file, output_folder, batch_size=19):
    with open(combined_file, 'r') as f:
        data = json.load(f)

    total_jobs = len(data)
    batch_count = (total_jobs + batch_size - 1) // batch_size  # How many batches we need

    for batch_num in range(batch_count):
        # Create a batch from the original data and an output file
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_jobs)
        batch_data = data[start_idx:end_idx]
        output_file = os.path.join(output_folder, f"combined_data_s{batch_num + 1}.json")
        with open(output_file, 'w') as out_file:
            json.dump(batch_data, out_file, indent=4)

        print(f"Saved {len(batch_data)} jobs to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <cif_folder>")
        sys.exit(1)

    cif_folder = sys.argv[1]
    output_file = os.path.join(cif_folder, "combined_data.json")

    if not os.path.isdir(cif_folder):
        print(f"Error: {cif_folder} is not a valid directory.")
        sys.exit(1)

    process_cif_folder(cif_folder, output_file)