# -------------------------------------------------------------------
from typing import Dict, List
# G0 AF3 internal confidence
# -------------------------------------------------------------------
AF3_NUM = [
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_rna_pTM_list_value",
]
AF3_CAT: List[str] = []

# -------------------------------------------------------------------
# G1 Evolutionary signal
# -------------------------------------------------------------------
EVO_NUM = [
    "log10_rna_msa_size",
    "n_unique_rna_families",
    "n_rbp_templates",
    "n_protein_templates",
]
EVO_CAT: List[str] = [
    "rna_family",
]

# -------------------------------------------------------------------
# G2 Complexity / composition
# -------------------------------------------------------------------
COMPLEXITY_NUM = [
    "NumberRNAs",
    "NumberProteins",
    "rna_total_length",
    "protein_total_length",
    "total_complex_length",
    "protein_to_rna_length_ratio",
    "max_rna_fraction_of_total_rna",
]
COMPLEXITY_CAT: List[str] = []

# -------------------------------------------------------------------
# G3 RNA intrinsic properties
# -------------------------------------------------------------------
RNA_INTRINSIC_NUM = [
    "RNA_SequenceComplexity",
    "RNA_Shannon_Entropy_K1",
    "RNA_MFE_Value",
    "RNA_Ensemble_Diversity",
    "RNA_Centroid_Energy",
    "agg_rna_sequence_complexity",
    "agg_rna_shannon_entropy_k2",
]
RNA_INTRINSIC_CAT: List[str] = []

RNA_STRUCTURE_NUM = [
    "num_stems",
    "num_hairpin_loops",
    "num_internal_loops",
    "num_multibranch_loops",
    "num_pseudoknots",
]
RNA_STRUCTURE_CAT: List[str] = []

BIOPHYS_NUM = [
    "log10_kd",
]
BIOPHYS_CAT: List[str] = []

# -------------------------------------------------------------------
# G4 Interface features
# -------------------------------------------------------------------
INTERFACE_AGREEMENT_NUM = [
    "pred_ProteinRNAInterfaceArea",
    "pred_ProteinRNAInterfaceRatio",
]
INTERFACE_AGREEMENT_CAT: List[str] = []

GLOBAL_INTERFACE_NUM = [
    "n_unique_rna_interface_residues_global",
    "n_unique_protein_interface_residues_global",
    "rna_interface_coverage_global",
    "interface_contact_distance_mean",
    "interface_contact_distance_median",
    "frac_contacts_lt_3_5A",
    "interface_rna_motif_length_sum",
    "interface_rna_motif_coverage_of_total_rna",
    "interface_frac_contacts_to_C",
]
GLOBAL_INTERFACE_CAT: List[str] = []

CHAIN_INTERFACE_NUM = [
    "chain_n_unique_rna_interface_residues",
    "chain_interface_contact_distance_mean",
    "chain_frac_contacts_lt_3_5A",
    "chain_interface_rna_motif_length_sum",
]
CHAIN_INTERFACE_CAT: List[str] = []

# -------------------------------------------------------------------
# G5 Experimental setup + chemistry / environment
# -------------------------------------------------------------------
CHEM_NUM = [
    "n_ions_listed",
    "n_ligands_listed",
    "ionic_strength",
    "peg_percent",
    "buffer_m",
    "has_peg",
]
CHEM_CAT = [
    "peg_concentration_category",
    "buffer_category",
]

EXP_NUM = [
    "exp_pH",
    "exp_Temp",
    "XRayResolution",
    "is_xray",
    # "is_post_cutoff",
]
EXP_CAT: List[str] = []