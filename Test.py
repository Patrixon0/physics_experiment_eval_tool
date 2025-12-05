"""
Small demo hub showing how to call the tools from code.

Run one of:
  python Test.py --example function_fitter
  python Test.py --example function_fitter_settings
  python Test.py --example geraden_fit
"""

import argparse
import sys
from pathlib import Path

# ensure the package root is on sys.path when running from this file directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from function_fitter import run_function_fitter  # type: ignore
from function_fitter_config import FunctionFitterConfig  # type: ignore
from geraden_fit import geraden_fit  # type: ignore


def example_function_fitter():
    """Launch the function fitter with a simple sine demo CSV."""
    cfg = FunctionFitterConfig(
        csv_path="fitter/function_fitter_test_data/demo_sine.csv",
        function_expr="a * np.sin(2*np.pi*f*x + phi) + c",
        x_label="t [s]",
        y_label="signal [1]",
        show_total_curve=True,
        show_sum_pos=True,
        show_sum_neg=True,
        show_target_curve=False,
        load_settings=False,  # do not override the demo parameters
    )
    run_function_fitter(cfg)


def example_function_fitter_with_saved_settings():
    """
    Load the standard Pb demo data, but also auto-load saved settings
    (e.g. Pb_energy_fit1.settings.json) if present.
    """
    cfg = FunctionFitterConfig(
        csv_path="fitter/function_fitter_test_data/Pb_energy.csv",
        load_settings=True,
    )
    run_function_fitter(cfg)


def example_geraden_fit():
    """
    Minimal call into the linear fit helper with a made-up CSV.
    Adjust the path to point to your own CSV with two columns (x, y).
    """
    csv_path = REPO_ROOT / "fitter" / "function_fitter_test_data" / "demo_sine.csv"
    geraden_fit(
        str(csv_path),
        title="Sine demo as scatter",
        x_label="t [s]",
        y_label="signal [1]",
        save=False,
        linear_fit=True,
        focus_point=False,
        plot_y_inter=True,
    )


EXAMPLES = {
    "function_fitter": example_function_fitter,
    "function_fitter_settings": example_function_fitter_with_saved_settings,
    "geraden_fit": example_geraden_fit,
}


def main():
    parser = argparse.ArgumentParser(description="Run small demos for the eval tools.")
    parser.add_argument(
        "--example",
        choices=sorted(EXAMPLES.keys()),
        default="function_fitter",
        help="Which demo to launch.",
    )
    args = parser.parse_args()

    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
