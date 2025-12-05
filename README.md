# physics-experiment-eval-tool

Helper scripts for common physics lab evaluations:
- Function fitter (interactive PyQtGraph GUI for arbitrary f(x) with sliders)
- Linear fit / regression helpers (`geraden_fit`)
- Gaussian error propagation (`gauss_fehlerfortpflanzung`)
- Scientific error rounding and table helpers

## Quickstart

1) Install deps (PyQt5, pyqtgraph, numpy, pandas, sympy, matplotlib):  
   `pip install -r requirements.txt`

2) Run the interactive function fitter with a demo config:  
   `python function_fitter.py --config config_1`

   - Configs live in `function_fitter_config.py` (see `FunctionFitterConfig`).  
   - Add your own by pointing `csv_path` to a 2-col CSV (x,y) and optionally setting `function_expr`, axis labels, and visibility toggles.  
   - Saved UI settings (`*.settings.json`) next to your CSV can be auto-loaded when `load_settings=True`.

3) See code usage examples in `Test.py`:  
   - `python Test.py --example function_fitter` (sine demo)  
   - `python Test.py --example function_fitter_settings` (loads Pb demo + saved settings)  
   - `python Test.py --example geraden_fit` (linear regression call)

4) Headless / automation:  
   `python function_fitter.py --config config_1 --no-gui --optimize`  
   prints χ²/χ²_red to the terminal without opening the window.

## Config-driven usage

- Function fitter: create a `FunctionFitterConfig` in `function_fitter_config.py`, then run
  `python function_fitter.py --config your_config_name` or call `run_function_fitter(your_config)` from your own code.
- Linear fit: import and call `geraden_fit(...)`; see `geraden_fit_config.py` for configurable parameters or `Test.py` for a minimal snippet.

## Demos / data

- Example CSVs live in `fitter/function_fitter_test_data/` (`demo_sine.csv`, `Pb_energy.csv`, ...).
- Output artifacts (settings JSON, LaTeX tables, PNG exports) are written next to the input CSV unless overridden in your call.

## Synthetic data generator

Use `fitter/generate_synthetic_mixtures.py` to create noisy datasets built from random sums of gaussians/sines/lines.  
Each CSV gets a companion `.meta.json` with the ground-truth expression to plug into your configs.
