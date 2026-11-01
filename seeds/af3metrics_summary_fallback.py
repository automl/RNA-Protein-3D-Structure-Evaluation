"""AF3 confidence metrics of the seed databases (all five models per seed), with a per-model fallback.

The shipped af3Metrics_new.py stops at the first model whose pred row has no RNAMotifLength (the model 1-4 rows cloned by
seeds/populate_models.py have none: get_chain_pair_with_longest_motif raises), and it writes nothing for a model unless
the PAE part succeeds. The published seed DBs hold summary confidences (pTM/ipTM/ranking/clash/disorder) for all models
and pLDDT/PAE only where the PAE part succeeded. This script runs the same main loop with the functions of
evaluation/af3Metrics_new.py; when the PAE part raises, it writes the summary-confidence columns only (pLDDT/PAE stay
NULL), which reproduces the published seed databases. Run it in place of af3Metrics_new.py (see seeds/README.md):

    python seeds/af3metrics_summary_fallback.py <af folder of one seed>
"""
import glob, json, os, sys

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder = sys.argv[1]
sys.path[:0] = [repo, os.path.join(repo, "evaluation")]
import af3Metrics_new as m

SUMMARY = ["af3_protein_pTM", "af3_rna_pTM", "af3_protein_ipTM", "af3_rna_ipTM",
           "af3_fraction_disordered", "af3_has_clash", "af3_ranking_score"]
iptm_files = glob.glob(os.path.join(folder, "*_summary_confidences_[0-4].json"))
pae_files = glob.glob(os.path.join(folder, "*_full_data_[0-4].json"))
n_full = n_summary = n_skip = 0
for jf in sorted(iptm_files):
    iptm = m.get_af3Metrics_paths_iptm(folder, jf)
    if not iptm:
        n_skip += 1; continue
    cif = m.derive_cif_file_name(os.path.basename(jf))
    match = [f for f in pae_files if m.derive_cif_file_name(os.path.basename(f)) == cif]
    pae = None
    if match:
        try:
            pae = m.get_af3Metrics_paths_pae(cif, match[0], iptm)
        except Exception as e:
            print(f"PAE part failed for {cif}: {type(e).__name__}: {e}")
    if pae:
        m.insert_data_to_db(cif, iptm, pae); n_full += 1
    else:
        m.database_methods.update_or_insert(
            "pred_protein_rna", SUMMARY,
            (json.dumps(iptm["protein_ptm"]), json.dumps(iptm["rna_ptm"]), json.dumps(iptm["protein_iptm"]),
             json.dumps(iptm["rna_iptm"]), iptm["fraction_disordered"], iptm["has_clash"], iptm["af3_ranking_score"]),
            condition=f"FileName = '{cif}'")
        n_summary += 1
print(f"full metrics: {n_full}; summary-only (PAE part failed): {n_summary}; skipped: {n_skip}")
