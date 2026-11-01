import os
import sys
import json
import glob
import re
import shutil
import argparse
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from collections import defaultdict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import logging

logging.basicConfig(level=logging.ERROR)

RNA_BOND_CUTOFF = 2.5  # Å; the O3'(i)->P(i+1) phosphodiester bond is ~1.6 Å when intact


def _atom(res, *names):
    for n in names:
        if res.has_id(n):
            return res[n]
    return None


def _is_rna_residue(res):
    # RNA (ribose) residues carry the 2'-OH oxygen; DNA and protein residues do not
    return res.has_id("O2'") or res.has_id("O2*")


def rna_chain_breaks(cif_file, bond_cutoff=RNA_BOND_CUTOFF, max_gap=0):
    """List discontinuities in the RNA chains of a deposited CIF.

    A break is reported when consecutive RNA residues either skip more than
    ``max_gap`` residue numbers (missing residues) or fail the covalent
    O3'(i)->P(i+1) bond-distance test. Because the phosphodiester bond is ~1.6 Å
    when intact and >>3 Å when broken, the geometry decision is binary and no
    borderline distance needs manual inspection.

    ``max_gap`` sets how many consecutive missing residues are tolerated before a
    chain counts as broken: ``max_gap=1`` keeps a single missing residue (the case
    the old P--P>10 Å method sent to manual review) and flags gaps of two or more;
    ``max_gap=0`` is strict (any missing residue breaks); a large value flags only
    genuine backbone breaks. Returns [] when the RNA chains are continuous.
    """
    structure = MMCIFParser(QUIET=True).get_structure("s", cif_file)
    breaks = []
    for chain in structure[0]:
        residues = [r for r in chain if _is_rna_residue(r)]
        for prev, cur in zip(residues, residues[1:]):
            missing = cur.id[1] - prev.id[1] - 1
            if missing > max_gap:                                  # too many missing residues
                breaks.append((chain.id, prev.id[1], cur.id[1], "gap", missing))
                continue
            if missing > 0:                                        # tolerated gap: no bond expected
                continue
            o3, p = _atom(prev, "O3'", "O3*"), _atom(cur, "P")
            if o3 is None or p is None:                            # incomplete backbone
                breaks.append((chain.id, prev.id[1], cur.id[1], "missing_atom", None))
            elif (o3 - p) > bond_cutoff:                           # broken phosphodiester bond
                breaks.append((chain.id, prev.id[1], cur.id[1], "broken", round(o3 - p, 2)))
    return breaks


def _residue_count(seq):
    """Number of residues in an assembled sequence string, counting each (MOD) token as one."""
    return len(re.sub(r'\([A-Z0-9]+\)', 'X', seq))


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
        
        if best_match_idx != -1:
            if entity_poly_type[best_match_idx] == 'polypeptide(L)':
                return "proteinChain"
            elif entity_poly_type[best_match_idx] == 'polyribonucleotide':
                return "rnaSequence"
            elif entity_poly_type[best_match_idx] == 'polydeoxyribonucleotide':
                return "dnaSequence"
    
    return None

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

# allowed_ligands = {CCD_ADP, CCD_ATP, CCD_AMP, CCD_GTP, CCD_GDP, CCD_FAD, CCD_NAD, CCD_NAP, CCD_NDP, CCD_HEM, CCD_HEC,
#                    CCD_PLM, CCD_OLA, CCD_MYR, CCD_CIT, CCD_CLA, CCD_CHL, CCD_BCL, CCD_BCB}

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
            return True
        # else:
        #     print(f"Residue '{resname}' not found in sequence: {sequence}")
    return False

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
                return None, "modification", resname
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
                return None, "modification", resname

        for chain_id, residues in sequences.items():
            # Count residues, not string characters: each modified residue is a 5-char "(XXX)"
            # token, so len(joined) would over-count and let sub-4-residue chains slip through.
            if isinstance(residues, list):
                n_res = len(residues)
                sequences[chain_id] = ''.join(residues)
            else:
                n_res = _residue_count(residues)
            if n_res < 4:
                return None, "length", n_res
    # for chain, sequence in sequences.items():
    #     length = len(sequence)
    #     if length < 4:
    #         print(f"Error {cif_file}, chain {chain}: sequence is too short: {len(sequence)}.")
    #         return None
    #     else:
    return sequences, None, None

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

def save_combined_json(sequences, output_file, batch_size=30):
    combined_json = []
    for base_name, seqs in sequences.items():
        job = {
            "name": f"{base_name}_s1",
            "modelSeeds": [1],
            "sequences": seqs
        }
        combined_json.append(job)

    # with open(output_file, 'w') as json_file:
    #     json.dump(combined_json, json_file, indent=4)
    # Split combined_json into batches of batch_size jobs per JSON file
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


def _write_classification_report(folder_path, report):
    """Write one txt file listing the PDB IDs in each outcome bucket."""
    path = os.path.join(folder_path, "cif_classification.txt")
    order = ["passed", "modification", "length", "disconnected"]
    labels = {
        "passed": "PASSED all tests",
        "modification": "FAILED - invalid modification / ligand",
        "length": "FAILED - size (chain < 4 residues, or total residues > max)",
        "disconnected": "FAILED - disconnected RNA backbone",
    }
    total = sum(len(report[k]) for k in order)
    n_pass = len(report["passed"])
    n_fail = total - n_pass
    with open(path, "w") as f:
        f.write(f"# {total} CIF files classified: {n_pass} passed, {n_fail} failed\n")
        f.write(f"#   modification={len(report['modification'])}  "
                f"length={len(report['length'])}  "
                f"disconnected={len(report['disconnected'])}\n\n")
        for key in order:
            entries = sorted(set(report[key]))
            f.write(f"# {labels[key]} (n={len(entries)})\n")
            for pdb_id, detail in entries:
                f.write(f"{pdb_id}\t{detail}\n" if detail else f"{pdb_id}\n")
            f.write("\n")
    print(f"Classified {total} files: {n_pass} passed, {n_fail} failed "
          f"(modification={len(report['modification'])}, length={len(report['length'])}, "
          f"disconnected={len(report['disconnected'])}). Report: {path}")


def process_cif_folder(folder_path, output_file, max_residues=5120, batch_size=30):
    all_sequences = {}
    report = {"passed": [], "modification": [], "length": [], "disconnected": []}

    def _sort_out(cif_path, subfolder):
        dest_dir = os.path.join(folder_path, subfolder)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(cif_path, os.path.join(dest_dir, os.path.basename(cif_path)))

    for cif_file in glob.glob(os.path.join(folder_path, "*.cif")):
        base_name = os.path.basename(cif_file).split('.')[0]  # Get the base name of the file (without extension)
        parsed_sequences, reason, detail = parse_cif_sequence(cif_file)  # reason: 'modification'|'length'|None

        # An independent test failed (invalid modification/ligand, or a too-short chain): move the
        # file into the matching subfolder and record the specific reason in the report (not printed).
        if reason == "modification":
            _sort_out(cif_file, "modification")
            report["modification"].append((base_name, f"modification {detail}"))
            continue
        if reason == "length":
            _sort_out(cif_file, "length")
            report["length"].append((base_name, f"length {detail} nt (< 4)"))
            continue

        # Would otherwise have passed, but a deposited RNA chain is discontinuous: set it aside
        # rather than submit a broken structure we could not reliably score later.
        breaks = rna_chain_breaks(cif_file)
        if breaks:
            _sort_out(cif_file, "disconnected")
            report["disconnected"].append((base_name, f"RNA chain break(s): {breaks}"))
            continue

        # Total-residue (token) cap: reject complexes whose combined polymer length exceeds the cap.
        # Counts protein + RNA (+ DNA) residues together; for RNA-only inputs this is the RNA length.
        # Disabled with max_residues <= 0.
        if max_residues and max_residues > 0:
            total_res = sum(_residue_count(seq) for seq in parsed_sequences.values())
            if total_res > max_residues:
                _sort_out(cif_file, "length")
                report["length"].append((base_name, f"residues {total_res} (> {max_residues} max)"))
                continue

        report["passed"].append((base_name, ""))
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

    save_combined_json(all_sequences, output_file, batch_size=batch_size)
    _write_classification_report(folder_path, report)


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
    ap = argparse.ArgumentParser(
        description="Convert deposited CIFs in a folder to AlphaFold3 input JSON; sort failures "
                    "into modification/, length/ and disconnected/ subfolders and write "
                    "cif_classification.txt.")
    ap.add_argument("cif_folder", help="Folder containing *.cif files")
    ap.add_argument("--maxResidues", type=int, default=5120, metavar="N",
                    help="Maximum total polymer residues per complex (protein + RNA together, or RNA "
                         "only for RNA-only inputs). Applied by default at 5120 without needing the "
                         "flag; pass --maxResidues N to change it, or --maxResidues 0 to disable. "
                         "Complexes exceeding the cap are sorted into length/.")
    ap.add_argument("--jobsPerFile", type=int, default=30, metavar="N",
                    help="Number of jobs per output JSON file (combined_data_1.json, "
                         "combined_data_2.json, ...). Default 30; pass e.g. --jobsPerFile 100 "
                         "for more.")
    args = ap.parse_args()

    if not os.path.isdir(args.cif_folder):
        ap.error(f"'{args.cif_folder}' is not a valid directory")
    if args.jobsPerFile < 1:
        ap.error("--jobsPerFile must be at least 1")

    output_file = os.path.join(args.cif_folder, "combined_data.json")
    process_cif_folder(args.cif_folder, output_file, max_residues=args.maxResidues,
                       batch_size=args.jobsPerFile)