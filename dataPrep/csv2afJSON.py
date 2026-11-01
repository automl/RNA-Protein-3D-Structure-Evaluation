import csv
import json
import ast  # To safely parse stringified lists
import re

def parse_simple_sequences_from_csv(csv_file):
    jobs = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        # print("CSV columns:", reader.fieldnames)

        for row in reader:
            job_name = row["exp_db_id"]

            def clean_seq_string(s):
                s = s.strip()

                # Remove surrounding quotes or brackets (if any)
                s = re.sub(r'^[\[\("\']+', '', s)  # remove leading [("'
                s = re.sub(r'[\]\)"\']+$', '', s)  # remove trailing ])"'

                return s.strip()

            # Safely parse string representation of lists
            # rna_seqs = ast.literal_eval(row["RNASequence"])
            rna_seqs = [clean_seq_string(s) for s in row["RNASequence"].split(',') if s.strip()]
            protein_seqs = [clean_seq_string(s) for s in row["ProteinSequence"].split(',') if s.strip()]

            # protein_seqs = ast.literal_eval(row["ProteinSequence"])

            seq_entries = []

            for pseq in protein_seqs:
                seq_entries.append({
                    "proteinChain": {
                        "sequence": pseq,
                        "modifications": [],
                        "count": 1
                    }
                })

            for rseq in rna_seqs:
                seq_entries.append({
                    "rnaSequence": {
                        "sequence": rseq,
                        "modifications": [],
                        "count": 1
                    }
                })

            job = {
                "name": job_name,
                "modelSeeds": [1],
                "sequences": seq_entries
            }

            jobs.append(job)

    return jobs

def parse_single_sequences_from_csv(csv_file):
    jobs = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file, delimiter=',')

        for row in reader:
            # Only include rows where NumberRNAs == 1
            if row.get("NumberRNAs") != "1":
                continue

            job_name = row["exp_db_id"]
            print("job_name",job_name)

            def clean_seq_string(s):
                s = s.strip()
                s = re.sub(r'^[\[\("\']+', '', s)
                s = re.sub(r'[\]\)"\']+$', '', s)
                return s.strip()

            rna_seqs = [clean_seq_string(s) for s in row["RNASequence"].split(',') if s.strip()]
            protein_seqs = [clean_seq_string(s) for s in row["ProteinSequence"].split(',') if s.strip()]

            seq_entries = []

            for pseq in protein_seqs:
                seq_entries.append({
                    "proteinChain": {
                        "sequence": pseq,
                        "modifications": [],
                        "count": 1
                    }
                })

            for rseq in rna_seqs:
                seq_entries.append({
                    "rnaSequence": {
                        "sequence": rseq,
                        "modifications": [],
                        "count": 1
                    }
                })

            job = {
                "name": job_name,
                "modelSeeds": [1],
                "sequences": seq_entries
            }

            jobs.append(job)

    return jobs


def save_jobs_to_json(jobs, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(jobs, json_file, indent=4)
    print(f"✅ Saved {len(jobs)} jobs to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python dataPrep/csv2afJSON.py pred_protein_rna.csv out.json")
        sys.exit(1)
    save_jobs_to_json(parse_single_sequences_from_csv(sys.argv[1]), sys.argv[2])
