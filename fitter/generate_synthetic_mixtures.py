"""
Utility to generate synthetic datasets made from sums of different functions
plus noise, along with helper metadata for configs.

Run:
    python generate_synthetic_mixtures.py --out-dir fitter/function_fitter_test_data --samples 5
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


FUNCS = ["gaussian", "sine", "line"]


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sine(x, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * x + phase) + offset


def line(x, m, b):
    return m * x + b


@dataclass
class Component:
    kind: str
    params: dict

    def expression(self, idx: int) -> str:
        """Return a Python expression string using unique parameter names with index idx."""
        if self.kind == "gaussian":
            return f"a{idx}*np.exp(-0.5*((x-mu{idx})/sigma{idx})**2)"
        if self.kind == "sine":
            return f"a{idx}*np.sin(2*np.pi*f{idx}*x + phi{idx}) + c{idx}"
        if self.kind == "line":
            return f"m{idx}*x + b{idx}"
        raise ValueError(f"Unknown kind {self.kind}")

    def param_values(self, idx: int) -> dict:
        mapping = {}
        for key, val in self.params.items():
            mapped_key = key
            if key in ("amp", "a"):
                mapped_key = f"a{idx}"
            elif key in ("mu",):
                mapped_key = f"mu{idx}"
            elif key in ("sigma",):
                mapped_key = f"sigma{idx}"
            elif key in ("freq",):
                mapped_key = f"f{idx}"
            elif key in ("phase",):
                mapped_key = f"phi{idx}"
            elif key in ("offset", "c"):
                mapped_key = f"c{idx}"
            elif key in ("m",):
                mapped_key = f"m{idx}"
            elif key in ("b",):
                mapped_key = f"b{idx}"
            mapping[mapped_key] = val
        return mapping


def random_component(x_span: Tuple[float, float]) -> Component:
    kind = random.choice(FUNCS)
    lo, hi = x_span
    if kind == "gaussian":
        amp = random.uniform(0.5, 2.5)
        mu = random.uniform(lo, hi)
        sigma = random.uniform(0.2, max(0.5, (hi - lo) / 4))
        return Component(kind, {"amp": amp, "mu": mu, "sigma": sigma})
    if kind == "sine":
        amp = random.uniform(0.5, 2.0)
        freq = random.uniform(0.05, 0.2)
        phase = random.uniform(0, np.pi)
        offset = random.uniform(-0.5, 0.5)
        return Component(kind, {"amp": amp, "freq": freq, "phase": phase, "offset": offset})
    m = random.uniform(-1.0, 1.0)
    b = random.uniform(-1.0, 1.0)
    return Component(kind, {"m": m, "b": b})


def synthesize_sample(idx: int, out_dir: Path, n_components: int, n_points: int = 400, sigma_noise: float = 0.1):
    rng = np.random.default_rng(seed=idx * 1337 + n_components)
    x = np.linspace(0, 10, n_points)
    comps: List[Component] = []
    y = np.zeros_like(x)
    for i in range(n_components):
        comp = random_component((x.min(), x.max()))
        comps.append(comp)
        if comp.kind == "gaussian":
            y += gaussian(x, comp.params["amp"], comp.params["mu"], comp.params["sigma"])
        elif comp.kind == "sine":
            y += sine(x, comp.params["amp"], comp.params["freq"], comp.params["phase"], comp.params["offset"])
        elif comp.kind == "line":
            y += line(x, comp.params["m"], comp.params["b"])

    noise = rng.normal(0, sigma_noise, size=y.shape)
    y_noisy = y + noise
    y_err = np.full_like(y, sigma_noise)

    csv_path = out_dir / f"synthetic_{n_components:02d}_comp_{idx:02d}.csv"
    pd.DataFrame({"x": x, "y": y_noisy, "y_err": y_err}).to_csv(csv_path, index=False)

    expr_parts = []
    params_combined = {}
    for i, comp in enumerate(comps):
        expr_parts.append(comp.expression(i))
        params_combined.update(comp.param_values(i))

    meta = {
        "components": [asdict(c) for c in comps],
        "expression": " + ".join(expr_parts),
        "true_params": params_combined,
        "csv": str(csv_path),
    }
    meta_path = out_dir / f"synthetic_{n_components:02d}_comp_{idx:02d}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # also write a ready-to-load settings slot (fit1) for convenience
    def make_component_settings(comp, i):
        vals = comp.param_values(i)
        ranges = {}
        for k, v in vals.items():
            span = max(abs(v) * 2.0, 1.0)
            ranges[k] = (v - span, v + span)
        return {
            "type": "pos",
            "visible": True,
            "expr": comp.expression(i),
            "param_names": list(vals.keys()),
            "params": vals,
            "ranges": ranges,
            "exponents": {},
            "mantissas": vals,
            "label": comp.kind,
            "color": None,
        }

    settings_base = {
        "function_expr": meta["expression"],
        "param_names": list(params_combined.keys()),
        "x_label": "x",
        "y_label": "y",
        "param_defaults": {},
        "x_range": None,
        "global_range_min": -10.0,
        "global_range_max": 10.0,
        "global_param_ranges": {},
        "components": [make_component_settings(comp, i) for i, comp in enumerate(comps)],
        "has_error_columns": False,
        "y_errors_in_column": "y_err",
        "settings_slot": "fit1",
    }
    settings_path = out_dir / f"synthetic_{n_components:02d}_comp_{idx:02d}_fit1.settings.json"
    settings_path.write_text(json.dumps(settings_base, indent=2))

    # rough/randomized ranges/settings for testing optimization robustness
    rough_components = []
    for i, comp in enumerate(comps):
        vals = comp.param_values(i)
        param_names = list(vals.keys())
        params = {}
        ranges = {}
        mantissas = {}
        rng = np.random.default_rng(seed=idx * 2024 + i)
        for name, val in vals.items():
            # randomize start around truth with moderate noise
            jitter = rng.normal(scale=max(abs(val) * 0.3, 0.2))
            start = val + jitter
            span = max(abs(val) * 2.5, 2.0)
            lo = start - span
            hi = start + span
            params[name] = start
            mantissas[name] = start
            ranges[name] = (lo, hi)
        rough_components.append({
            "type": "pos",
            "visible": True,
            "expr": comp.expression(i),
            "param_names": param_names,
            "params": params,
            "ranges": ranges,
            "exponents": {},
            "mantissas": mantissas,
            "label": f"{comp.kind}_rough",
            "color": None,
        })

    rough_settings = dict(settings_base)
    rough_settings["components"] = rough_components
    rough_settings["settings_slot"] = "rough"
    rough_path = out_dir / f"synthetic_{n_components:02d}_comp_{idx:02d}_rough.settings.json"
    rough_path.write_text(json.dumps(rough_settings, indent=2))

    return csv_path, meta


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mixed-function datasets.")
    parser.add_argument("--out-dir", type=Path, default=Path("fitter/function_fitter_test_data"))
    parser.add_argument("--samples", type=int, default=5, help="number of datasets per component count")
    parser.add_argument("--max-components", type=int, default=5, help="maximum number of components")
    parser.add_argument("--sigma-noise", type=float, default=0.1, help="std dev of gaussian noise / y_err")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for n in range(1, args.max_components + 1):
        for s in range(args.samples):
            csv_path, meta = synthesize_sample(s, args.out_dir, n, sigma_noise=args.sigma_noise)
            summary.append(meta)
            print(f"[ok] wrote {csv_path} with expression: {meta['expression']}")

    # write a helper file listing all generated configs
    summary_path = args.out_dir / "synthetic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")
    print("Each .meta.json contains the expression string you can paste into function_fitter_config.py")


if __name__ == "__main__":
    main()
