import requests
import json
import time
import os
import urllib

# EBI InterProScan requires a contact email (no token/API key). Configure it via the INTERPRO_EMAIL
# environment variable instead of hardcoding a personal address.
INTERPRO_EMAIL = os.environ.get("INTERPRO_EMAIL", "").strip()

# https://colab.research.google.com/drive/1UKJ-5fY254ALs_nNBvggpwc3raMkNtzo?usp=sharing#scrollTo=UnU26tQ2bBFt
def extract_interpro_entries(uniprot_id):
    UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb"
    url =  '{}/{}.json'.format(UNIPROT_API_URL, uniprot_id)
    uniprot_results = json.load(urllib.request.urlopen(url))
    # sequence = uniprot_results['sequence']['value']

    interpro_entries = []
    if 'uniProtKBCrossReferences' in uniprot_results and isinstance(uniprot_results['uniProtKBCrossReferences'], list):
        for reference in uniprot_results['uniProtKBCrossReferences']:
            if reference.get('database') == 'InterPro':
                # Iterate over properties to find the relevant 'EntryName'
                for prop in reference.get('properties', []):
                    if prop.get('key') == 'EntryName':
                        entry_name = prop.get('value')
                        if entry_name:
                            interpro_entries.append({
                                # "database": reference.get('database'),
                                # "id": reference.get('id'),
                                "FamilyName": entry_name
                            })

    return interpro_entries

# Example:
uniprot_id = 'Q15650'
print(extract_interpro_entries(uniprot_id))

# https://www.ebi.ac.uk/interpro/api/static_files/swagger/#/Entry/get_entry
# https://gustavo-salazar.github.io/ProteinFamiliesTalks/InterProAPI.html#/25
# https://www.youtube.com/watch?v=8i3yszapfRo&t=1559s
def get_protein_family_from_sequence(sequence, pdb_id):
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    if not INTERPRO_EMAIL:
        raise RuntimeError(
            "ProteinFamily lookup needs a contact email for the EBI InterProScan API. "
            "Set the INTERPRO_EMAIL environment variable to your email address."
        )
    url_submit = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'email': INTERPRO_EMAIL, 'sequence': sequence}

    response_submit = requests.post(url_submit, headers=headers, data=data)
    if response_submit.status_code != 200:
        raise Exception(f"Failed to submit sequence: {response_submit.status_code}, {response_submit.text}")
    job_id = response_submit.text
    print(f"Job submitted with ID: {job_id}")

    url_status = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/status/{job_id}"

    while True:
        status = requests.get(url_status).text
        print(f"Job status: {status}")
        if status == 'FINISHED':
            break
        elif status in ['RUNNING', 'PENDING']:
            time.sleep(30)

    url_results = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/result/{job_id}/json"
    response_results = requests.get(url_results)
    if response_results.status_code != 200:
        raise Exception(f"Failed to retrieve results: {response_results.status_code}, {response_results.text}")

    payload = response_results.json()

    protein_families = []

    for result in payload.get('results', []):
        for match in result.get('matches', []):
            signature = match.get('signature', {})
            entry = signature.get('entry', {})
            if entry and entry.get('type') == 'FAMILY':
                family_name = entry.get('name')
                print(family_name)
                protein_families.append(family_name)

    if not protein_families:
        # Optionally dump the raw payload for debugging if PROTEIN_FAMILY_DEBUG_DIR is set; otherwise
        # just return None (a machine-specific default path used to fail on other machines).
        debug_dir = os.environ.get("PROTEIN_FAMILY_DEBUG_DIR", "").strip()
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"{pdb_id}.json"), 'w') as json_file:
                    json.dump(payload, json_file, indent=4)
            except OSError:
                pass
        return None

    return protein_families


def get_protein_families_per_sequences(sequences, pdb_id):
    """
    Get protein families for each chain sequence. Returns a nested list: one list of family names per sequence,
    e.g. [["ArgoL1", "PAZ_dom", ...], ["CRISPR_Csm3_endoribonuclease"]] for two chains.
    """
    if not sequences:
        return []
    result = []
    for i, seq in enumerate(sequences):
        label = f"{pdb_id}_chain{i}"
        families = get_protein_family_from_sequence(seq, label)
        # Use [None] (JSON [null]) per chain when no families found, so DB has one entry per chain
        result.append(families if families else [None])
    return result


    # To save raw JSON file if no detected family_name:
    # output_folder = 'data'
    # output_filename = os.path.join(output_folder, f"{job_id}.json")
    # with open(output_filename, 'w') as json_file:
    #     json.dump(payload, json_file, indent=4)

# Example
# sequence = "FKPPPRPDFGTSGRTIKLQANFFEMDIPKIDIYHYELDIKPEKCPRRVNREIVEHMVQHFKTQIFGDRKPVFDGRKNLYTAMPLPIGRDKVELEVTLPGEKDRIFKVSIKWVSCVSLQALHDALSGRLPSVPFETIQALDVVMRHLPSMRYTPVGRSFFTASEGCSNPLGGGREVWFGFHQSVRPSLWKMMLNIDVSATAFYKAQPVIEFVCEVLDFKSIEEQQKPLTDSQRVKFTKEIKGLKVEITHCGKRKYRVCNVTRRPASHQTFPLQQESGQTVECTVAQYFKDRHKLVLRYPHLPCLQVGQEQKHTYLPLEVCNIVAGQRCIKKLTDNQTSTMIRATARSAPDRQEEISKLMRSADFNTDPYVREFGIMVKDEMTDVTGRVLQPPSILYGGRNKAIATPVQGVWDMRNKQFHTGIEIKVWAIACFAPQRQCTEVHLKSFTEQLRKISRDAGMPIQGQPCFCKYAQGADSVEPMFRHLKNTYAGLQLVVVILPGKTPVYAEVKRVGDTVLGMATQCVQMKNVQRTTPQTLSNLCLKINVKLGGVNNILLPQGRPPVFQQPVIFLGADVTHPPGKKPSIAAVVGSMDAHPNRYCATVRVQQHRQEIIQDLAAMVRELLIQFYKSTRFKPTRIIFYRDGVSEGQFQQVLHHELLAIREACIKLEKDYQPGITFIVVQKRHHTRLFCTDKNERVGKSGNIPAGTTVDTKITHPTEFDFYLCSHAGIQGTSRPSHYHVLWDDNRFSSDELQILTYQLCHTYVRCTRSVSIPAPAYYAHLVAFRARYHLHQALAKAVQVHQDTLRTMYFA"
# protein_families = get_protein_family_from_sequence(sequence, "c150")
# print(protein_families)

# def extract_protein_family_from_json(json_filename):
#     with open(json_filename, 'r') as json_file:
#         payload = json.load(json_file)
#
#     protein_families = []
#
#     for result in payload.get('results', []):
#         for match in result.get('matches', []):
#             signature = match.get('signature', {})
#             entry = signature.get('entry', {})
#             if entry and entry.get('type') == 'FAMILY':
#                 family_name = entry.get('name')
#                 print(family_name)
#                 protein_families.append(family_name)
#
#     return protein_families
#
# def process_json_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             json_filepath = os.path.join(folder_path, filename)
#             families = extract_protein_family_from_json(json_filepath)
#             print(f"File: {filename}, Protein Families: {families}")
#
# folder_path = 'results_folder'
# process_json_folder('data')
