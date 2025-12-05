"""
Evaluate robustness of the global optimizer on synthetic datasets.

For each meta/settings pair in a directory:
  - compute chi^2 using the reference fit1 settings (target)
  - compute chi^2 after starting from rough settings and running global optimize
  - compare ratios to see how often optimization lands near the target solution
"""

from pathlib import Path
import json
from typing import List, Tuple
from datetime import datetime

from function_fitter import run_function_fitter  # type: ignore
from function_fitter_config import FunctionFitterConfig  # type: ignore


def find_pairs(base_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """Return list of (csv, fit1_settings, rough_settings) triples."""
    triples = []
    for fit1 in base_dir.glob("synthetic_*_fit1.settings.json"):
        rough = fit1.with_name(fit1.name.replace("_fit1.settings", "_rough.settings"))
        if not rough.exists():
            continue
        stem = fit1.name.replace("_fit1.settings.json", "")
        csv = base_dir / f"{stem}.csv"
        if csv.exists():
            triples.append((csv, fit1, rough))
    return sorted(triples)


def chi2_from_settings(csv_path: Path, settings_path: Path, optimize: bool) -> float:
    cfg = FunctionFitterConfig(
        csv_path=str(csv_path),
        has_error_columns=False,
        y_error_column="y_err",
        load_settings=True,
        settings_slot="fit1" if "fit1" in settings_path.name else "rough",
    )
    # override default slot naming by renaming the file to match slot inside run
    # run_function_fitter reads settings based on <csv_stem>_<slot>.settings.json
    # here we just ensure slot matches the filename suffix
    chi = run_function_fitter(cfg, show_window=False, run_optimize=optimize, return_result=True)
    if chi is None:
        return float("inf")
    return chi[0]  # chi^2


def bucket_counts(diffs: List[float]):
    buckets = {
        "<=10%": 0,
        "<=30%": 0,
        "<=70%": 0,
        ">70%": 0,
    }
    for d in diffs:
        if d <= 0.10:
            buckets["<=10%"] += 1
        elif d <= 0.30:
            buckets["<=30%"] += 1
        elif d <= 0.70:
            buckets["<=70%"] += 1
        else:
            buckets[">70%"] += 1
    total = len(diffs)
    return {k: (v, 100.0 * v / total if total else 0.0) for k, v in buckets.items()}


def main():
    base_dir = Path("fitter/function_fitter_test_data")
    pairs = find_pairs(base_dir)
    if not pairs:
        print("No synthetic settings found.")
        return

    ratios = []
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Eval run at {timestamp}")

    for csv_path, fit1, rough in pairs:
        target_chi = chi2_from_settings(csv_path, fit1, optimize=False)
        opt_chi = chi2_from_settings(csv_path, rough, optimize=True)
        if opt_chi <= target_chi:
            ratio = 0.0
        else:
            ratio = (opt_chi - target_chi) / target_chi if target_chi != 0 else float("inf")
        ratios.append(ratio)
        line = f"{csv_path.name}: target chi^2={target_chi:.4g}, optimized chi^2={opt_chi:.4g}, ratio={ratio:.3f}"
        print(line)
        lines.append(line)

    buckets = bucket_counts(ratios)
    print("\nBucket results (count, %):")
    lines.append("\nBucket results (count, %):")
    for label, (count, pct) in buckets.items():
        bucket_line = f"  {label}: {count} ({pct:.1f}%)"
        print(bucket_line)
        lines.append(bucket_line)

    out_path = base_dir / "optimizer_eval_report.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
