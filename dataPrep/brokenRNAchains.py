"""Flag RNA chain breaks in deposited structures (dataset curation, Appendix attrition).

A break is a distance above BREAK_THRESHOLD (10 A) between the phosphorus atoms of consecutive
nucleotides of an RNA chain. Entries whose breaks are all between 10 and 12 A are listed for
manual inspection. Run:  python dataPrep/brokenRNAchains.py <folder with deposited .cif files>
"""
import os
import sys
import pymol
from pymol import cmd
from pathlib import Path

# Threshold in Ångström for detecting a break between P atoms in adjacent nucleotides
BREAK_THRESHOLD = 10.0


def detect_rna_breaks_in_cif(file_path):
    cmd.reinitialize()
    cmd.load(file_path)

    result = []
    for obj in cmd.get_object_list():
        # Get all chains in the object
        chains = cmd.get_chains(obj)
        for chain in chains:
            # Check if this chain is RNA (contains at least one RNA residue)
            rna_residues = cmd.count_atoms(f"{obj} and chain {chain} and resn A+U+G+C+I")
            if rna_residues == 0:
                continue  # Not an RNA chain

            # Select only the phosphate atoms (P) for connectivity
            sele = f"{obj} and chain {chain} and name P and resn A+U+G+C+I"
            model = cmd.get_model(sele)

            # Sort by residue number
            atoms = sorted(model.atom, key=lambda a: a.resi_number)
            for i in range(1, len(atoms)):
                a1 = atoms[i - 1]
                a2 = atoms[i]
                dist = ((a1.coord[0] - a2.coord[0]) ** 2 +
                        (a1.coord[1] - a2.coord[1]) ** 2 +
                        (a1.coord[2] - a2.coord[2]) ** 2) ** 0.5
                if dist > BREAK_THRESHOLD:
                    result.append((os.path.basename(file_path), chain, a1.resi, a2.resi, round(dist, 2)))
    return result


def scan_folder_for_rna_breaks(cif_folder):
    all_breaks = []
    for cif_file in Path(cif_folder).glob("*.cif"):
        breaks = detect_rna_breaks_in_cif(str(cif_file))
        if breaks:
            all_breaks.extend(breaks)
    return all_breaks


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python dataPrep/brokenRNAchains.py <folder with deposited .cif files>")
        sys.exit(1)
    pymol.finish_launching(['pymol', '-qc'])  # quiet and no GUI
    folder = sys.argv[1]
    broken_chains = scan_folder_for_rna_breaks(folder)

    if broken_chains:
        pdb_breaks = {}
        pdb_manual = {}
        for entry in broken_chains:
            pdb_id = os.path.splitext(entry[0])[0]
            distance = entry[4]
            if pdb_id not in pdb_breaks:
                pdb_breaks[pdb_id] = []
            pdb_breaks[pdb_id].append(distance)

        print(len(pdb_breaks))

        # Now filter for manual check: at least one break in 10-12, none above 12
        for pdb_id, distances in pdb_breaks.items():
            in_10_12 = any(10 <= d <= 12 for d in distances)
            over_12 = any(d > 12 for d in distances)
            if in_10_12 and not over_12:
                pdb_manual[pdb_id] = [d for d in distances if 10 <= d <= 12]

        print("PDB IDs with disconnected RNA chains:")
        print(", ".join(pdb_breaks.keys()))

        print("\nBreak distances for each PDB ID:")
        for pdb_id, distances in pdb_breaks.items():
            print(f"{pdb_id}: {', '.join(str(d) for d in distances)}")

        # Print manual check PDB IDs as a comma-separated list (no quotes)
        if pdb_manual:
            print("\nPDB IDs with ONLY break distances between 10 and 12 Å (manual check recommended):")
            print(", ".join(pdb_manual.keys()))
            for pdb_id, distances in pdb_manual.items():
                formatted = ", ".join(f"{d:.2f}" for d in distances)
                print(f"{pdb_id}: {formatted}")
        else:
            print("\nNo PDB IDs with only break distances between 10 and 12 Å.")
    else:
        print("No disconnected RNA chains detected.")
