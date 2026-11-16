"""Rebuild both prepared CSVs from the original NPPmalfunctions XLSX files.

Usage:
    python scripts/prepare_data.py --dataset-root "path/to/Malfunctions"

Source: https://github.com/SFStefenon/NPPmalfunctions
"""
from pathlib import Path
import argparse
import pandas as pd


def prepare(dataset_root: Path, output_root: Path) -> None:
    files = sorted(dataset_root.glob("*/*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No scenario .xlsx files found in {dataset_root}")
    frames = []
    reference_columns = None
    for path in files:
        scenario = path.parent.name
        try:
            scenario_id = int(scenario.split(" - ")[0])
        except (ValueError, IndexError):
            raise ValueError(f"Unrecognized scenario folder: {scenario}")
        # ID 13 is explicitly 'Reserved for future', not a confirmed fault.
        if scenario_id == 13:
            continue
        frame = pd.read_excel(path)
        if reference_columns is None:
            reference_columns = list(frame.columns)
        elif list(frame.columns) != reference_columns:
            raise ValueError(f"Inconsistent sensor columns: {path}")
        frame["Scenario"] = scenario
        frame["Scenario_ID"] = scenario_id
        frame["Label"] = 0 if scenario_id == 0 else 1
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    # Exploratory preprocessing decision: remove the shared first fault row.
    prepared = combined.loc[
        ~((combined["Label"] == 1) & (combined["TIME"] == 0))
    ].reset_index(drop=True)
    revised = prepared.loc[
        ~prepared["Scenario_ID"].isin([5, 8, 14])
    ].reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_root / "npp_prepared.csv", index=False)
    revised.to_csv(output_root / "npp_revised.csv", index=False)
    print(f"Prepared: {prepared.shape}; revised: {revised.shape}")
    print("NOTE: labels indicate scenario membership, not verified fault onset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("prepared_data"))
    args = parser.parse_args()
    prepare(args.dataset_root, args.output_root)
