"""Reading pred_protein_rna.RNA_interface_placement.

The column holds the placement class of every protein-RNA interface (evaluation/interface_rna_pose.py), in the
pair notation of RBP_DockQ: [{"A:C": "right_protein_site_wrong_rna_pose"}, {"B:C": "shifted_nearby"}, ...], with the
chain ids of the DB (predicted chain naming) and snake_case class labels. Analyses that need one class per RNA chain
derive it here; the per-RNA-chain summaries by main protein partner (largest native interface) are computed from the
pose tables by publication_figures/placement_per_rna_chain.py.
"""
import json

# placement classes from best to worst (interface_rna_pose.PLACEMENTS, snake_case)
ORDER = ["correct_interface", "same_place_duplex_slid", "right_protein_site_wrong_rna_pose",
         "protein_binding_region_mispredicted", "same_place_same_orientation", "same_place_rna_turned",
         "shifted_nearby", "different_region"]
RANK = {c: i for i, c in enumerate(ORDER)}


def interface_placement(raw):
    """{"P:R": class} for every interface; also accepts the earlier per-RNA-chain notation {"R": class}."""
    if raw in (None, ""):
        return {}
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {}
    if isinstance(v, dict):
        return dict(v)
    out = {}
    for item in v or []:
        if isinstance(item, dict):
            out.update(item)
    return out


def best_per_rna_chain(raw):
    """{RNA chain: best class over its protein-RNA interfaces} (the rule of the decision-tree labels)."""
    best = {}
    for pair, cls in interface_placement(raw).items():
        rna = pair.split(":")[-1]
        if cls in RANK and (rna not in best or RANK[cls] < RANK[best[rna]]):
            best[rna] = cls
    return best
