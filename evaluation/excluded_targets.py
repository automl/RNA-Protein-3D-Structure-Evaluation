"""
Targets removed from the pre-/post-cutoff comparison (evaluation/af3_training_membership.py).

Both entries were listed in both the pre-cutoff (b2021) and the post-cutoff (a2021) set:
  6FQL  released 2018-05-09 -> in the AF3 training set; removed from the post-cutoff set
  7PMQ  deposited 2021-09-02 but released 2022-09-14 -> not in the AF3 training set; removed
        from the pre-cutoff set
The rows were deleted from data/compareDatabases/post_pre/a2021_b2021_protein_rna/{a2021,b2021}.db
and database/rbpDatabase.db; scripts that read other copies (e.g. the seed databases) filter here.
"""
POST_EXCLUDED = {"6FQL"}
PRE_EXCLUDED = {"7PMQ"}
