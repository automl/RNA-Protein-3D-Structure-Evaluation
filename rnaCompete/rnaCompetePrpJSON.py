import csv
import json
import random
from collections import defaultdict

def parse_sequences_from_csv(csv_file):
    sequences = {}

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file, delimiter=',')

        data_by_protein = defaultdict(list)
        for row in reader:
            data_by_protein[row["ProteinId"]].append(row)

        for protein_id, rows in data_by_protein.items():
            maxprobes = [r for r in rows if r["Class"] == "maxprobe"]
            minprobes = [r for r in rows if r["Class"] == "minprobe"]

            if not maxprobes or not minprobes:
                continue  # skip if no max or minprobe available

            # Pick best maxprobe based on highest z_score + binding
            best_maxprobe = max(
                maxprobes,
                key=lambda r: float(r["RNAMotif_z_score"]) + float(r["BindingIntensity"])
            )

            # Extract base name from PDBId
            pdb_id_full = best_maxprobe["PDBId"]
            first_part = pdb_id_full.split("max")[0]
            last_four_digits = pdb_id_full[-4:]
            base_name = first_part + last_four_digits

            protein_seq = best_maxprobe["ProteinSequence"].strip('[]"')
            rna_seq_max = best_maxprobe["RNASequence"].strip('[]"')

            # Build protein and maxprobe RNA entries
            protein_entry = {
                "proteinChain": {
                    "sequence": protein_seq,
                    "modifications": [],
                    "count": 1
                }
            }

            rna_entry_max = {
                "rnaSequence": {
                    "sequence": rna_seq_max,
                    "modifications": [],
                    "count": 1
                }
            }

            # Sort minprobes by RNAMotif_z_score (ascending)
            sorted_minprobes = sorted(minprobes, key=lambda r: float(r["RNAMotif_z_score"]))

            # Create three entries with progressively more minprobes
            for n_min in range(1, 4):
                # Select minprobes based on RNAMotif_z_score
                selected_minprobes = sorted_minprobes[:n_min]  # Take the n_min lowest scoring minprobes

                # Convert to JSON entries
                min_entries = []
                for r in selected_minprobes:
                    rna_seq = r["RNASequence"].strip('[]"')
                    min_entries.append({
                        "rnaSequence": {
                            "sequence": rna_seq,
                            "modifications": [],
                            "count": 1
                        }
                    })

                # Build job name
                job_name = f"{base_name}Max1Min{n_min}"

                # Add to dictionary
                sequences[job_name] = [protein_entry, rna_entry_max] + min_entries

    return sequences



def save_combined_json(sequences, output_file):
    combined_json = []
    for job_name, seqs in sequences.items():
        job = {
            "name": job_name,
            "modelSeeds": [1],
            "sequences": seqs
        }
        combined_json.append(job)

    batch_size = 800
    total_jobs = len(combined_json)
    batch_count = (total_jobs + batch_size - 1) // batch_size

    for batch_num in range(batch_count):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_jobs)
        batch_data = combined_json[start_idx:end_idx]

        output_file_batch = output_file.replace('.json', f'_{batch_num + 1}.json')

        with open(output_file_batch, 'w') as json_file:
            json.dump(batch_data, json_file, indent=4)

        print(f"✅ Saved {len(batch_data)} jobs to {output_file_batch}")

# Example usage:
if __name__ == "__main__":
    import os
    results = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "protein_rna_RNAcompete_v3", "results")
    csv_file = os.path.join(results, "exp_protein_rna.csv")
    output_json = os.path.join(results, "combined_batch.json")

    sequences = parse_sequences_from_csv(csv_file)

    # Print each job to terminal
    for base_name, seqs in sequences.items():
        print(f"\n▶ {base_name}_protein_s1")
        for s in seqs:
            print(json.dumps(s, indent=4))

    # Save to batched JSON files
    save_combined_json(sequences, output_json)
