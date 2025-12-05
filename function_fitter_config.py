from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionFitterConfig:
    """Configuration for the interactive function fitter."""

    csv_path: str
    has_error_columns: bool = False
    y_error_column: Optional[str] = None
    settings_slot: Optional[str] = "fit1"
    function_expr: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    show_total_curve: Optional[bool] = True
    show_sum_pos: Optional[bool] = True
    show_sum_neg: Optional[bool] = True
    show_target_curve: Optional[bool] = True
    load_settings: bool = True


# Example configs that can be referenced via --config
config_1 = FunctionFitterConfig(
    csv_path="fitter/function_fitter_test_data/Pb_energy.csv",
    has_error_columns=False,
    function_expr=None,  # use default gauss
    x_label="Energie [keV]",
    y_label="Intensität [1]",
    load_settings=True,
)

config_no_settings = FunctionFitterConfig(
    csv_path="fitter/function_fitter_test_data/Pb_energy.csv",
    has_error_columns=False,
    load_settings=False,
)
