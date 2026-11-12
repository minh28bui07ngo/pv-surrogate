"""
Convert old simulation outputs into the three new targets: SS, SC, LCC.

Old outputs can be either
  (a) hourly format: 2 columns, each cell an 8760-point array
        "Electricity:Facility", "ElectricityProduced:Facility"
  (b) annual format: 4 columns
        "Electricity:Facility", "ElectricityProduced:Facility",
        "E_self", "Annual_grid_cost(VND)"

New outputs (one row per sample, same row order as the old file):
  SS  = E_self / Electricity:Facility
  SC  = E_self / ElectricityProduced:Facility
  LCC = initial PV cost + sum_{y=1..20} annual_grid_cost / (1 + 0.10)^y
        initial PV cost = sum(PV1..PV8) / 198 * 2_500_000 * 1.15
  (same definitions as besos_kriging.py)

LCC needs the PV capacities, so each output file is paired with its input file.

Usage:
  python convert_outputs.py --pair inputs_test.pkl outputs_test.pkl
  python convert_outputs.py --pair inputs_train.pkl outputs_train.pkl \
                            --pair inputs_test.pkl  outputs_test.pkl  --out-dir new_data

Creates, for every pair, next to the old output (or in --out-dir):
  <outputs_name>_new.pkl   DataFrame with columns SS, SC, LCC
  <outputs_name>_new.csv   same data, readable in Excel
The input files are left untouched (the Kriging script still needs them).
"""
import argparse
import os

import numpy as np
import pandas as pd

# ---- constants, identical to besos_kriging.py -------------------------------
PV_COLS = slice(32, 40)          # PV1..PV8 in the input DataFrame
W_PER_PANEL_AREA = 198
COST_PER_AREA = 2_500_000
COST_FACTOR = 1.15
LIFETIME = 20
DISCOUNT_RATE = 0.10

FAC, PROD = "Electricity:Facility", "ElectricityProduced:Facility"
SELF, GRID = "E_self", "Annual_grid_cost(VND)"
NEW_NAMES = ["SS", "SC", "LCC"]


def tariff(hours):
    """Hourly tariff in VND/kWh, same rule as get_electricity_rate()."""
    h = np.asarray(hours, dtype=float)
    return np.where(
        (h >= 22) | (h < 4), 1918.0,
        np.where(((h >= 9.5) & (h < 11.5)) | ((h >= 13.5) & (h < 17)), 3152.0, 5422.0),
    )


def annual_from_hourly(row):
    load = np.asarray(row[FAC], dtype=float)
    pv = np.asarray(row[PROD], dtype=float)
    if len(load) != len(pv):
        raise ValueError("Facility and Produced arrays differ in length.")
    e_self = np.minimum(load, pv).sum()
    grid_kwh = np.maximum(load - pv, 0.0) / (1000 * 3600)   # J -> kWh
    cost = (grid_kwh * tariff(np.arange(len(load)) % 24)).sum()
    return load.sum(), pv.sum(), e_self, cost


def convert(inputs, outputs):
    if len(inputs) != len(outputs):
        raise ValueError(f"Row mismatch: inputs {len(inputs)} vs outputs {len(outputs)}")
    inputs = inputs.reset_index(drop=True)
    outputs = outputs.reset_index(drop=True)

    if GRID in outputs.columns and SELF in outputs.columns:          # annual format
        fac, prod, e_self, grid = (outputs[c].to_numpy(float) for c in (FAC, PROD, SELF, GRID))
    elif FAC in outputs.columns and PROD in outputs.columns:         # hourly format
        a = np.array([annual_from_hourly(r) for _, r in outputs.iterrows()])
        fac, prod, e_self, grid = a.T
    else:
        raise ValueError(f"Unrecognised output columns: {list(outputs.columns)}")

    pv_total = inputs.iloc[:, PV_COLS].to_numpy(float).sum(axis=1)
    initial_cost = pv_total / W_PER_PANEL_AREA * COST_PER_AREA * COST_FACTOR
    annuity = sum(1 / (1 + DISCOUNT_RATE) ** y for y in range(1, LIFETIME + 1))

    new = pd.DataFrame({
        "SS": e_self / fac,
        "SC": e_self / prod,
        "LCC": initial_cost + grid * annuity,
    })
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True,
                    metavar=("INPUTS_PKL", "OUTPUTS_PKL"),
                    help="an input file and its matching old output file (repeatable)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write new files (default: next to each old output file)")
    args = ap.parse_args()

    for in_path, out_path in args.pair:
        new = convert(pd.read_pickle(in_path), pd.read_pickle(out_path))

        base = os.path.splitext(os.path.basename(out_path))[0] + "_new"
        folder = args.out_dir or os.path.dirname(os.path.abspath(out_path))
        os.makedirs(folder, exist_ok=True)
        pkl, csv = os.path.join(folder, base + ".pkl"), os.path.join(folder, base + ".csv")
        new.to_pickle(pkl)
        new.to_csv(csv, index=False)

        print(f"\n{out_path} -> {pkl}")
        print(f"                 {csv}")
        print(new.describe().loc[["count", "min", "mean", "max"]].to_string())
        bad = int(((new.SS > 1) | (new.SC > 1) | (new[NEW_NAMES].isna().any(axis=1))).sum())
        print(f"rows with SS>1, SC>1 or NaN: {bad}")


if __name__ == "__main__":
    main()
