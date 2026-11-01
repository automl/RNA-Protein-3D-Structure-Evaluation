# RNA-binding domain table and test case

Copied with the author's permission from **michauhl/rbd-annotation** (https://github.com/michauhl/rbd-annotation,
commit `1efb655`, 2026-09-08). Used by `database/annotateRBDs.py`.

| File | Source in rbd-annotation |
|---|---|
| `rbd_parent_infos.tsv` | `data/rbp2go.selected_new_rbds.reform.added.parent_infos.txt`: the RBP2GO "Selected and new RBDs" InterPro entries plus manual additions (1,000 entries), each with its InterPro parent entry (`parent_id`, `parent_short`, `parent_type`) |
| `test/test_protein_seqs.interproscan_results.tsv` | `data/test_protein_seqs.interproscan_results.out`: InterProScan 5 TSV output for the author's test proteins |
| `test/test_protein_seqs.interpro_rbd_domains.expected.tsv` | `data/test_protein_seqs.interpro_rbd_domains.txt`: the author's RBD hits for that output (one row per hit) |

`python database/annotateRBDs.py --test` checks that the adapted rules reproduce the expected hits.
