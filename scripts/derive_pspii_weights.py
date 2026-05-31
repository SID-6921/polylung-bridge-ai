import argparse
import json
from pathlib import Path

import pandas as pd


def min_max_norm(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive PSPII weights from extracted cytokine data.")
    parser.add_argument("--input", required=True, help="CSV with columns: polymer, TNFa, IL1b, IL5, IL6, MIP2a")
    parser.add_argument("--output", default="evidence/public/pd2/pspii_weights_final.json")
    parser.add_argument("--allow-placeholder", action="store_true", help="Allow placeholder zero-only inputs")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"polymer", "TNFa", "IL1b", "IL5", "IL6", "MIP2a"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"Missing columns: {sorted(missing)}")

    marker_cols = ["TNFa", "IL1b", "IL5", "IL6", "MIP2a"]

    if not args.allow_placeholder:
        marker_sum = df[marker_cols].abs().sum().sum()
        if float(marker_sum) == 0.0:
            raise ValueError(
                "Input appears to be placeholder data (all cytokine values are zero). "
                "Provide real extracted values from WebPlotDigitizer or pass --allow-placeholder explicitly."
            )

    norm_df = df.copy()
    for c in marker_cols:
        norm_df[c] = min_max_norm(norm_df[c])

    norm_df["pspii"] = norm_df[marker_cols].mean(axis=1)
    out = {
        row["polymer"]: round(float(row["pspii"]), 4)
        for _, row in norm_df[["polymer", "pspii"]].iterrows()
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote PSPII weights to {out_path}")


if __name__ == "__main__":
    main()
