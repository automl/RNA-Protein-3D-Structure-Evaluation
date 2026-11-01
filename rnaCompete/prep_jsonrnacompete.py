import json
import os

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "protein_rna_RNAcompete_v3", "results")
reference_file = os.path.join(_RESULTS, "rnaCompete_v3_exp1_random_filtered.json")
target_file = os.path.join(_RESULTS, "combined_batch_1.json")
output_file = os.path.join(_RESULTS, "missingMinMax.json")

# Load files
with open(reference_file, "r") as f:
    reference_data = json.load(f)

with open(target_file, "r") as f:
    target_data = json.load(f)

# Collect names from reference file
reference_names = {
    sample["name"]
    for sample in reference_data
    if isinstance(sample, dict) and "name" in sample
}

# Keep only samples whose names are not in reference file
filtered_data = [
    sample
    for sample in target_data
    if sample.get("name") not in reference_names
]

# Save result
with open(output_file, "w") as f:
    json.dump(filtered_data, f, indent=4)

print(
    f"Removed {len(target_data) - len(filtered_data)} samples. "
    f"Remaining: {len(filtered_data)}"
)

# import json
# import pandas as pd
#
# csv_file = "data/protein_rna/pred_protein_rna.csv"
# json_file = "data/protein_rna_RNAcompete_v3/results/rnaCompete_v3_exp1_random.json"
# output_json = "data/protein_rna_RNAcompete_v3/results/rnaCompete_v3_exp1_random_filtered.json"
#
# # Read CSV IDs
# df = pd.read_csv(csv_file)
#
# csv_ids = set(
#     df["exp_db_id"]
#     .dropna()
#     .astype(str)
#     .str.strip()
#     .str.lower()
# )
#
# # Read JSON samples
# with open(json_file, "r") as f:
#     json_data = json.load(f)
#
# # Keep only samples whose name exists in CSV
# filtered_data = []
# removed_names = []
#
# for sample in json_data:
#     sample_name = sample.get("name", "").strip()
#
#     if sample_name.lower() in csv_ids:
#         filtered_data.append(sample)
#     else:
#         removed_names.append(sample_name)
#
# # Save filtered JSON
# with open(output_json, "w") as f:
#     json.dump(filtered_data, f, indent=4)
#
# print(f"Original samples: {len(json_data)}")
# print(f"Kept samples: {len(filtered_data)}")
# print(f"Removed samples: {len(removed_names)}")
#
# # Optional: show first few removed IDs
# print("\nFirst 20 removed IDs:")
# for name in removed_names[:20]:
#     print(name)