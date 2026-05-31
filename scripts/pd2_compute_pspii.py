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
    parser = argparse.ArgumentParser(description="Compute PSPII from extracted cytokine values")
    parser.add_argument("--input", required=True, help="Path to extracted CSV")
    parser.add_argument("--group", default="PS_microplastic", help="Group to prioritize for PSPII reporting")
    parser.add_argument("--output-json", default="C:/Users/nanda/Downloads/PD2_pspii_weights_final.json")
    parser.add_argument("--output-csv", default="C:/Users/nanda/Downloads/PD2_pspii_debug.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"polymer", "TNFa", "IL1b", "IL5", "IL6", "MIP2a", "group"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    marker_cols = ["TNFa", "IL1b", "IL5", "IL6", "MIP2a"]
    marker_sum = df[marker_cols].abs().sum().sum()
    if float(marker_sum) == 0.0:
        raise ValueError("All cytokine values are zero. Fill real extracted values first.")

    norm_df = df.copy()
    for c in marker_cols:
        norm_df[c] = min_max_norm(norm_df[c])
    norm_df["pspii"] = norm_df[marker_cols].mean(axis=1)

    # Save full debug table.
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    norm_df.to_csv(out_csv, index=False)

    # Prefer chosen group if it exists; otherwise average by polymer.
    if args.group in set(norm_df["group"].astype(str)):
        subset = norm_df[norm_df["group"].astype(str) == args.group]
        result_series = subset.groupby("polymer")["pspii"].mean()
    else:
        result_series = norm_df.groupby("polymer")["pspii"].mean()

    out = {k: round(float(v), 4) for k, v in result_series.to_dict().items()}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Debug table: {out_csv}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
