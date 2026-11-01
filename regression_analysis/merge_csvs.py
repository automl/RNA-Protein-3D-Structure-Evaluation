#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


def combine_csvs(
    input_paths,
    output_path,
    add_source_column=True,
    source_from_filename=True,
):
    dfs = []

    for p in input_paths:
        p = Path(p)
        df = pd.read_csv(p)

        if add_source_column:
            if source_from_filename:
                df["source"] = p.stem
            else:
                df["source"] = str(p)

        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)

    # optional: ensure consistent column order
    combined = combined.reindex(sorted(combined.columns), axis=1)

    combined.to_csv(output_path, index=False)
    print(f"Saved combined CSV: {output_path}")
    print(f"Total rows: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")


if __name__ == "__main__":
    input_csvs = [
        "csvs4analysis_with_interface/combined/combined__multichain_complex.csv",
        "csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv",
    ]

    output_csv = "csvs4analysis_with_interface/combined/complex_level_all.csv"

    combine_csvs(input_csvs, output_csv)