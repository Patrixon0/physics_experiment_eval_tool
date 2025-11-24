import sys
import ast
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QScrollArea, QGroupBox, QLabel, QSlider, QDoubleSpinBox, QCheckBox, QLineEdit, QSpinBox
)
from PyQt5.QtCore import Qt

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter


DEFAULT_FUNCTION_EXPR = "amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)"


class GaussianFitUI(QWidget):
    def __init__(self, x, y, input_path: Path, x_err=None, y_err=None, has_error_columns: bool = False):
        super().__init__()

        self.x_data: np.ndarray = x.astype(float)
        self.y_data: np.ndarray = y.astype(float)
        self.has_error_columns = has_error_columns
        self.x_err = None if x_err is None else x_err.astype(float)
        self.y_err = None if y_err is None else y_err.astype(float)

        if self.x_err is not None and len(self.x_err) != len(self.x_data):
            raise ValueError("Length of x error column does not match x values.")
        if self.y_err is not None and len(self.y_err) != len(self.y_data):
            raise ValueError("Length of y error column does not match y values.")

        self.input_path = input_path
        self.components = []

        self.function_expr = DEFAULT_FUNCTION_EXPR
        self.param_names = self.extract_parameter_names(self.function_expr)
        self.safe_namespace = self.build_safe_namespace()
        self.formula_item = None
        self.param_default_settings = {}
        self.default_controls = {}
        self.x_range_settings = {}
        self.y_range_settings = {}
        self.show_sum_curves = True
        self.show_sum_pos = True
        self.show_sum_neg = True
        self.show_total_curve = True
        self.settings_slot = "fit1"
        self.range_dirty = True

        self.x_label = "Energie [keV]"
        self.y_label = "Intensität [1]"

        # High-resolution X grid for smooth curves
        self.reset_param_defaults()
        self.reset_x_range_defaults()
        self.reset_y_range_defaults()
        self.range_dirty = True
        N_high = max(2000, len(self.x_data) * 5)
        self.x_hr = np.linspace(self.x_range_min(), self.x_range_max(), N_high)

        self.init_ui()

    # --------------------------------------------------------
    # Helpers for dynamic functions
    # --------------------------------------------------------
    def build_safe_namespace(self):
        safe = {"np": np, "numpy": np, "pi": np.pi, "e": np.e}
        func_names = [
            "sin", "cos", "tan", "arcsin", "arccos", "arctan",
            "sinh", "cosh", "tanh", "exp", "log", "log10",
            "sqrt", "abs", "power"
        ]
        for name in func_names:
            safe[name] = getattr(np, name)
        return safe

    def split_mantissa_exponent(self, value: float):
        if value == 0:
            return 0.0, 0
        exp = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exp)
        return float(mantissa), exp

    def combine_mantissa_exponent(self, mantissa: float, exponent: int):
        return float(mantissa) * (10 ** int(exponent))

    def parse_color(self, color_str, fallback):
        try:
            c = pg.mkColor(color_str)
            return c
        except Exception:
            return pg.mkColor(fallback)

    def extract_parameter_names(self, expr: str):
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise SyntaxError(f"{exc}") from exc

        allowed = set(self.build_safe_namespace().keys())
        allowed.add("x")
        names = []

        class Visitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if node.id in allowed:
                    return
                if node.id not in names:
                    names.append(node.id)
                self.generic_visit(node)

        Visitor().visit(tree)
        return names

    def guess_param_defaults(self):
        param_values = {}
        param_ranges = {}
        param_exponents = {}

        x_min, x_max = float(self.x_data.min()), float(self.x_data.max())
        x_mean = float(self.x_data.mean())
        x_span = max(x_max - x_min, 1.0)
        y_min, y_max = float(self.y_data.min()), float(self.y_data.max())
        y_span = max(y_max - y_min, 1.0)

        for name in self.param_names:
            lower = name.lower()
            if lower in ("amp", "a", "amplitude"):
                val = y_max
                lo, hi = -2 * y_span, 2 * y_span
            elif lower in ("offset", "b", "c") or "offset" in lower:
                val = float(np.median(self.y_data))
                lo, hi = -2 * y_span, 2 * y_span
            elif any(token in lower for token in ("mu", "center", "x0", "c0")):
                val = x_mean
                lo, hi = x_min, x_max
            elif "sigma" in lower or "width" in lower or lower == "w":
                std = float(self.x_data.std() or 1.0)
                val = std
                lo, hi = max(std * 0.1, 1e-3), std * 5
            else:
                val = 1.0
                lo, hi = -2 * x_span, 2 * x_span

            param_values[name] = float(val)
            param_ranges[name] = (float(lo), float(hi))
            _, exp = self.split_mantissa_exponent(float(val))
            param_exponents[name] = exp

        return param_values, param_ranges, param_exponents

    def default_param_state(self):
        values, ranges, exponents = self.guess_param_defaults()
        return values, ranges, exponents

    def reset_param_defaults(self):
        values, _, exponents = self.default_param_state()
        self.param_default_settings = {}
        for name in self.param_names:
            mantissa, exp = self.split_mantissa_exponent(values[name])
            # prefer guessed exponent if non-zero
            chosen_exp = exponents.get(name, exp)
            mantissa = values[name] / (10 ** chosen_exp) if values[name] != 0 else 0.0
            self.param_default_settings[name] = {"mantissa": mantissa, "exp": chosen_exp}

    def ensure_default_settings(self):
        if not self.param_default_settings:
            self.reset_param_defaults()
            return
        values, _, exps = self.default_param_state()
        for name in self.param_names:
            if name not in self.param_default_settings:
                exp = exps.get(name, 0)
                mantissa = values[name] / (10 ** exp) if values[name] != 0 else 0.0
                self.param_default_settings[name] = {"mantissa": mantissa, "exp": exp}

    def reset_x_range_defaults(self):
        x_min_raw = float(self.x_data.min())
        x_max_raw = float(self.x_data.max())
        span = x_max_raw - x_min_raw
        pad = 0.05 * span if span != 0 else 0.05 * max(abs(x_min_raw), abs(x_max_raw), 1.0)
        x_min = x_min_raw - pad
        x_max = x_max_raw + pad
        m_min, e_min = self.split_mantissa_exponent(x_min if x_min != 0 else 1.0)
        m_max, e_max = self.split_mantissa_exponent(x_max if x_max != 0 else 1.0)
        self.x_range_settings = {
            "min_m": m_min if x_min != 0 else 1.0,
            "min_exp": e_min if x_min != 0 else 0,
            "max_m": m_max if x_max != 0 else 1.0,
            "max_exp": e_max if x_max != 0 else 0,
        }
        self.range_dirty = True

    def reset_y_range_defaults(self):
        y_min_raw = float(self.y_data.min())
        y_max_raw = float(self.y_data.max())
        span = y_max_raw - y_min_raw
        pad = 0.05 * span if span != 0 else 0.05 * max(abs(y_min_raw), abs(y_max_raw), 1.0)
        y_min = y_min_raw - pad
        y_max = y_max_raw + pad
        m_min, e_min = self.split_mantissa_exponent(y_min if y_min != 0 else 1.0)
        m_max, e_max = self.split_mantissa_exponent(y_max if y_max != 0 else 1.0)
        self.y_range_settings = {
            "min_m": m_min if y_min != 0 else 1.0,
            "min_exp": e_min if y_min != 0 else 0,
            "max_m": m_max if y_max != 0 else 1.0,
            "max_exp": e_max if y_max != 0 else 0,
        }
        self.range_dirty = True

    def x_range_min(self):
        return self.combine_mantissa_exponent(self.x_range_settings["min_m"], self.x_range_settings["min_exp"])

    def x_range_max(self):
        return self.combine_mantissa_exponent(self.x_range_settings["max_m"], self.x_range_settings["max_exp"])

    def y_range_min(self):
        return self.combine_mantissa_exponent(self.y_range_settings["min_m"], self.y_range_settings["min_exp"])

    def y_range_max(self):
        return self.combine_mantissa_exponent(self.y_range_settings["max_m"], self.y_range_settings["max_exp"])

    def add_positive_component(self):
        self._add_component("pos")

    def add_negative_component(self):
        self._add_component("neg")

    def _add_component(self, comp_type, param_values=None, ranges=None, exponents=None, mantissas=None, visible=True, update=True, color=None):
        defaults, default_ranges, default_exponents = self.default_param_state()
        self.ensure_default_settings()

        params = {}
        param_ranges = {}
        param_exponents = {}
        param_mantissas = {}
        for name in self.param_names:
            use_val = (param_values or {}).get(name, defaults[name])
            use_exp = (exponents or {}).get(name)
            if use_exp is None:
                use_exp = self.param_default_settings.get(name, {}).get("exp", default_exponents.get(name, 0))
            use_mantissa = (mantissas or {}).get(name)
            if use_mantissa is None:
                use_mantissa = self.param_default_settings.get(name, {}).get("mantissa")
                if use_mantissa is None:
                    factor = 10 ** use_exp if use_exp else 1
                    use_mantissa = use_val / factor if factor != 0 else use_val

            params[name] = float(self.combine_mantissa_exponent(use_mantissa, use_exp))
            base_range = (ranges or {}).get(name)
            if base_range is None:
                base_range = (-10.0, 10.0)
            param_ranges[name] = (float(base_range[0]), float(base_range[1]))
            param_exponents[name] = int(use_exp)
            param_mantissas[name] = float(use_mantissa)

        label = None
        if param_values and isinstance(param_values, dict):
            label = param_values.get("label")
        if label is None:
            label = f"{'+' if comp_type=='pos' else '-'} Component {len(self.components)}"

        if color is None and param_values and isinstance(param_values, dict):
            color = param_values.get("color")
        if color is None:
            color = "#c000c0" if comp_type == "pos" else "#555555"

        comp = {
            "type": comp_type,
            "params": params,
            "ranges": param_ranges,
            "exponents": param_exponents,
            "mantissas": param_mantissas,
            "label": label,
            "color": color,
            "visible": visible,
            "curve": None,
            "widgets": {}
        }
        self.components.append(comp)
        self._create_component_controls(comp)
        if update:
            self.update_curves()
        return comp

    # --------------------------------------------------------
    # UI setup
    # --------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("Function Plotter (PyQtGraph)")

        main_layout = QHBoxLayout(self)

        # ----------- plot area -----------
        self.plot_widget = pg.PlotWidget()

        self.param_text_item = pg.TextItem("", anchor=(0, 0))
        self.param_text_item.setPos(self.x_data.min(), self.y_data.max())
        self.plot_widget.addItem(self.param_text_item)

        self.legend = self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.setLabel("bottom", self.x_label)
        self.plot_widget.setLabel("left", self.y_label)

        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAntialiasing(True)

        pg.setConfigOptions(useOpenGL=False, antialias=True)

        main_layout.addWidget(self.plot_widget, stretch=3)

        self.plot = self.plot_widget.plot

        # main data
        self.data_curve = self.plot(
            self.x_data, self.y_data,
            pen=None,
            symbol="o",
            symbolBrush=(0, 0, 0),
            symbolSize=4,
            name="Data"
        )
        self.data_curve.setDownsampling(auto=False)
        self.data_curve.setClipToView(False)
        self.data_curve.setSkipFiniteCheck(True)

        self.error_bars = None
        if self.x_err is not None or self.y_err is not None:
            error_kwargs = {"x": self.x_data, "y": self.y_data, "beam": 0.0}
            if self.y_err is not None:
                error_kwargs["top"] = self.y_err
                error_kwargs["bottom"] = self.y_err
            if self.x_err is not None:
                error_kwargs["left"] = self.x_err
                error_kwargs["right"] = self.x_err
            self.error_bars = pg.ErrorBarItem(**error_kwargs, pen=pg.mkPen(color=(60, 60, 60), width=1))
            self.plot_widget.addItem(self.error_bars)

        # total fit curve
        self.fit_curve = self.plot(
            self.x_data,
            np.zeros_like(self.y_data),
            pen=pg.mkPen("r", width=2),
            name="Total"
        )
        self.fit_curve.setDownsampling(auto=False)
        self.fit_curve.setClipToView(False)
        self.fit_curve.setSkipFiniteCheck(True)
        self.fit_curve.setVisible(False)

        # sum + and sum – (shown positive)
        self.sum_pos_curve = self.plot(
            self.x_data, np.zeros_like(self.y_data),
            pen=pg.mkPen("b", width=2),
            name="Sum +"
        )
        self.sum_pos_curve.setDownsampling(auto=False)
        self.sum_pos_curve.setClipToView(False)
        self.sum_pos_curve.setSkipFiniteCheck(True)
        self.sum_pos_curve.setVisible(False)

        self.sum_neg_curve = self.plot(
            self.x_data, np.zeros_like(self.y_data),
            pen=pg.mkPen(color=(0, 0, 0), width=2, style=Qt.PenStyle.DashLine),
            name="Sum - (abs)"
        )
        self.sum_neg_curve.setDownsampling(auto=False)
        self.sum_neg_curve.setClipToView(False)
        self.sum_neg_curve.setSkipFiniteCheck(True)
        self.sum_neg_curve.setVisible(False)

        baseline = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(color=(0, 0, 0, 80), width=2)
        )
        self.plot_widget.addItem(baseline)

        self.update_formula_legend()

        # ----------- controls (separate window) -----------
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(5)

        # Function editor
        func_box = QGroupBox("Function f(x)")
        func_layout = QVBoxLayout(func_box)
        func_layout.setContentsMargins(6, 6, 6, 6)
        func_hint = QLabel("Use variable x and numpy (np) functions. Parameters become sliders.")
        func_hint.setWordWrap(True)
        self.function_edit = QLineEdit(self.function_expr)
        self.function_edit.setPlaceholderText("Example: amp * np.exp(-0.5*((x-mu)/sigma)**2) + c")
        self.btn_apply_function = QPushButton("Apply function")
        func_layout.addWidget(func_hint)
        func_layout.addWidget(self.function_edit)
        func_layout.addWidget(self.btn_apply_function)
        control_layout.addWidget(func_box)

        # Axes labels
        axes_box = QGroupBox("Axes labels")
        axes_layout = QVBoxLayout(axes_box)
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("x:"))
        self.x_label_edit = QLineEdit(self.x_label)
        x_row.addWidget(self.x_label_edit)
        axes_layout.addLayout(x_row)

        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("y:"))
        self.y_label_edit = QLineEdit(self.y_label)
        y_row.addWidget(self.y_label_edit)
        axes_layout.addLayout(y_row)
        control_layout.addWidget(axes_box)

        # Defaults for new components (mantissa * 10^exp)
        self.defaults_box = QGroupBox("Defaults for new components")
        self.defaults_layout = QVBoxLayout(self.defaults_box)
        self.defaults_layout.setContentsMargins(6, 6, 6, 6)
        self.defaults_layout.setSpacing(4)
        self.build_default_controls()
        control_layout.addWidget(self.defaults_box)

        # Settings slot
        slot_row = QHBoxLayout()
        slot_row.addWidget(QLabel("Settings slot:"))
        self.settings_slot_edit = QLineEdit(self.settings_slot)
        self.settings_slot_edit.setFixedWidth(180)
        self.settings_slot_edit.setPlaceholderText("e.g. default / fit1 / test")
        self.settings_slot_edit.setToolTip("Choose a slot name. Save/Load will use <csv_stem>_<slot>.settings.json")
        slot_row.addWidget(self.settings_slot_edit)
        slot_row.addStretch()
        control_layout.addLayout(slot_row)

        slot_hint = QLabel("Save/Load uses: <csv name>_<slot>.settings.json")
        slot_hint.setStyleSheet("color: #f0f0f0;")
        control_layout.addWidget(slot_hint)

        # Plot range controls
        self.range_box = QGroupBox("Plot range")
        self.range_layout = QVBoxLayout(self.range_box)
        self.range_layout.setContentsMargins(6, 6, 6, 6)
        self.range_layout.setSpacing(4)
        self.build_range_controls()
        control_layout.addWidget(self.range_box)

        # buttons row
        btn_row = QHBoxLayout()
        self.btn_add_pos = QPushButton("+ Component")
        self.btn_add_neg = QPushButton("- Component")
        btn_row.addWidget(self.btn_add_pos)
        btn_row.addWidget(self.btn_add_neg)
        control_layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.btn_save_clean = QPushButton("Save cleaned (neg)")
        self.btn_export_latex = QPushButton("Export LaTeX")
        btn_row2.addWidget(self.btn_save_clean)
        btn_row2.addWidget(self.btn_export_latex)
        control_layout.addLayout(btn_row2)

        visibility_box = QGroupBox("Curve visibility")
        vis_layout = QVBoxLayout(visibility_box)
        vis_layout.setContentsMargins(6, 6, 6, 6)
        vis_layout.setSpacing(2)
        self.chk_show_total = QCheckBox("Show total")
        self.chk_show_total.setChecked(True)
        self.chk_show_sum_pos = QCheckBox("Show sum +")
        self.chk_show_sum_pos.setChecked(True)
        self.chk_show_sum_neg = QCheckBox("Show sum -")
        self.chk_show_sum_neg.setChecked(True)
        vis_layout.addWidget(self.chk_show_total)
        vis_layout.addWidget(self.chk_show_sum_pos)
        vis_layout.addWidget(self.chk_show_sum_neg)
        control_layout.addWidget(visibility_box)

        # scroll area for component controls
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        control_layout.addWidget(self.scroll_area, stretch=1)

        self.comp_panel = QWidget()
        self.comp_layout = QVBoxLayout(self.comp_panel)
        self.comp_layout.setContentsMargins(0, 0, 0, 0)
        self.comp_layout.setSpacing(4)
        self.scroll_area.setWidget(self.comp_panel)

        # info label (parameter overview)
        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "QLabel { background-color: #f4f4f4; border: 1px solid #ccc; padding: 4px; font-family: monospace; color: #111; }"
        )
        self.info_label.setText(f"{self._info_label_prefix()}\nNo components yet.")
        self.info_label.setMinimumWidth(250)
        control_layout.addWidget(self.info_label)

        file_row = QHBoxLayout()
        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_load_settings = QPushButton("Load Settings")
        self.btn_save_png = QPushButton("Save plot as PNG")
        file_row.addWidget(self.btn_save_settings)
        file_row.addWidget(self.btn_load_settings)
        file_row.addWidget(self.btn_save_png)
        control_layout.addLayout(file_row)

        self.control_window = QWidget()
        self.control_window.setWindowTitle("Controls")
        cw_layout = QVBoxLayout(self.control_window)
        cw_layout.setContentsMargins(0, 0, 0, 0)
        cw_layout.addWidget(control_container)
        self.control_window.resize(420, 720)
        self.control_window.show()

        # connections
        self.btn_add_pos.clicked.connect(self.add_positive_component)
        self.btn_add_neg.clicked.connect(self.add_negative_component)
        self.btn_save_clean.clicked.connect(self.save_cleaned_negative)
        self.btn_export_latex.clicked.connect(self.export_latex_table)
        self.btn_save_settings.clicked.connect(self.save_settings)
        self.btn_load_settings.clicked.connect(self.load_settings)
        self.btn_save_png.clicked.connect(self.save_plot_png)
        self.btn_apply_function.clicked.connect(self.apply_function_expression)
        self.function_edit.returnPressed.connect(self.apply_function_expression)
        self.x_label_edit.editingFinished.connect(self.update_axis_labels_from_inputs)
        self.y_label_edit.editingFinished.connect(self.update_axis_labels_from_inputs)
        self.chk_show_total.stateChanged.connect(self.toggle_total_curve)
        self.chk_show_sum_pos.stateChanged.connect(self.toggle_sum_pos_curve)
        self.chk_show_sum_neg.stateChanged.connect(self.toggle_sum_neg_curve)

        self.apply_axis_labels()
        self.resize(1200, 720)
        self.control_window.move(self.x() + self.width(), self.y())

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def build_default_controls(self):
        if not hasattr(self, "defaults_layout"):
            return

        self._clear_layout(self.defaults_layout)
        self.default_controls = {}

        hint = QLabel("Mantissa × 10^exp is used as starting value for new components.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #f0f0f0;")
        self.defaults_layout.addWidget(hint)

        for name in self.param_names:
            row = QHBoxLayout()
            lab = QLabel(name)
            lab.setFixedWidth(50)
            row.addWidget(lab)

            mantissa_spin = QDoubleSpinBox()
            mantissa_spin.setDecimals(6)
            mantissa_spin.setRange(-1e9, 1e9)
            mantissa_spin.setFixedWidth(120)
            mantissa_spin.setValue(self.param_default_settings.get(name, {}).get("mantissa", 0.0))
            row.addWidget(mantissa_spin)

            exp_spin = QSpinBox()
            exp_spin.setRange(-18, 18)
            exp_spin.setPrefix("10^")
            exp_spin.setFixedWidth(120)
            exp_spin.setValue(self.param_default_settings.get(name, {}).get("exp", 0))
            row.addWidget(exp_spin)

            row.addStretch()
            self.defaults_layout.addLayout(row)

            def on_mantissa_changed(val, key=name):
                self.param_default_settings.setdefault(key, {})
                self.param_default_settings[key]["mantissa"] = float(val)

            def on_exp_changed(val, key=name):
                self.param_default_settings.setdefault(key, {})
                self.param_default_settings[key]["exp"] = int(val)

            mantissa_spin.valueChanged.connect(on_mantissa_changed)
            exp_spin.valueChanged.connect(on_exp_changed)

            self.default_controls[name] = {
                "mantissa": mantissa_spin,
                "exp": exp_spin,
            }

    def build_range_controls(self):
        if not hasattr(self, "range_layout"):
            return
        self._clear_layout(self.range_layout)

        hint = QLabel("Control x/y limits (mantissa × 10^exp). Defaults to data min/max.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #f0f0f0;")
        self.range_layout.addWidget(hint)

        def make_row(label, key_prefix, m_val, exp_val, target):
            row = QHBoxLayout()
            lab = QLabel(label)
            lab.setFixedWidth(40)
            lab.setStyleSheet("color: #f0f0f0;")
            row.addWidget(lab)

            m_spin = QDoubleSpinBox()
            m_spin.setDecimals(6)
            m_spin.setRange(-1e12, 1e12)
            m_spin.setFixedWidth(150)
            m_spin.setValue(m_val)
            row.addWidget(m_spin)

            exp_spin = QSpinBox()
            exp_spin.setRange(-18, 18)
            exp_spin.setPrefix("10^")
            exp_spin.setFixedWidth(120)
            exp_spin.setValue(exp_val)
            row.addWidget(exp_spin)

            def update_range():
                target[f"{key_prefix}_m"] = float(m_spin.value())
                target[f"{key_prefix}_exp"] = int(exp_spin.value())
                if target is self.x_range_settings:
                    self.update_x_grid()
                else:
                    self.update_view_ranges_only()
                self.range_dirty = True

            m_spin.valueChanged.connect(lambda _v: update_range())
            exp_spin.valueChanged.connect(lambda _v: update_range())

            row.addStretch()
            self.range_layout.addLayout(row)

        xlab = QLabel("X-range:")
        xlab.setStyleSheet("color: #f0f0f0;")
        self.range_layout.addWidget(xlab)
        make_row("Min", "min", self.x_range_settings["min_m"], self.x_range_settings["min_exp"], self.x_range_settings)
        make_row("Max", "max", self.x_range_settings["max_m"], self.x_range_settings["max_exp"], self.x_range_settings)

        ylab = QLabel("Y-range:")
        ylab.setStyleSheet("color: #f0f0f0;")
        self.range_layout.addWidget(ylab)
        make_row("Min", "min", self.y_range_settings["min_m"], self.y_range_settings["min_exp"], self.y_range_settings)
        make_row("Max", "max", self.y_range_settings["max_m"], self.y_range_settings["max_exp"], self.y_range_settings)

    # --------------------------------------------------------
    # Create per-component controls
    # --------------------------------------------------------
    def _create_component_controls(self, comp):
        index = len(self.components) - 1

        group = QGroupBox(f"{'+' if comp['type']=='pos' else '-'} Component {index}")
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(5, 5, 5, 5)
        vbox.setSpacing(3)

        header_layout = QHBoxLayout()
        chk_visible = QCheckBox("visible")
        chk_visible.setChecked(comp.get("visible", True))
        btn_delete = QPushButton("X")
        btn_delete.setFixedWidth(25)
        header_layout.addWidget(chk_visible)
        header_layout.addStretch()
        header_layout.addWidget(btn_delete)
        vbox.addLayout(header_layout)

        def make_row(label_text, param_key, mantissa_init, mantissa_min, mantissa_max, exp_init):
            row = QHBoxLayout()

            lab = QLabel(label_text)
            lab.setFixedWidth(50)
            row.addWidget(lab)

            min_box = QDoubleSpinBox()
            min_box.setDecimals(4)
            min_box.setRange(-1e12, 1e12)
            min_box.setFixedWidth(90)
            row.addWidget(min_box)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setFixedWidth(120)
            row.addWidget(slider)

            max_box = QDoubleSpinBox()
            max_box.setDecimals(4)
            max_box.setRange(-1e12, 1e12)
            max_box.setFixedWidth(90)
            row.addWidget(max_box)

            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setFixedWidth(110)
            row.addWidget(spin)

            exp_box = QSpinBox()
            exp_box.setRange(-18, 18)
            exp_box.setValue(exp_init)
            exp_box.setPrefix("10^")
            exp_box.setFixedWidth(110)
            row.addWidget(exp_box)

            vbox.addLayout(row)

            def current_exp():
                return exp_box.value()

            def update_slider_from_spin(val):
                lo = min_box.value()
                hi = max_box.value()
                if hi <= lo:
                    return
                t = (val - lo) / (hi - lo)
                slider.blockSignals(True)
                slider.setValue(int(1000 * t))
                slider.blockSignals(False)
                self._param_changed(comp, param_key, val, current_exp())

            def update_spin_from_slider(pos):
                lo = min_box.value()
                hi = max_box.value()
                if hi <= lo:
                    return
                t = pos / 1000.0
                val = lo + t * (hi - lo)
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
                self._param_changed(comp, param_key, val, current_exp())

            def update_ranges():
                lo = min_box.value()
                hi = max_box.value()
                if hi <= lo:
                    return

                spin.blockSignals(True)
                spin.setRange(lo, hi)
                spin.blockSignals(False)

                cur = spin.value()
                if cur < lo:
                    spin.setValue(lo)
                elif cur > hi:
                    spin.setValue(hi)

                update_slider_from_spin(spin.value())

                comp["ranges"][param_key] = (lo, hi)

            def on_exponent_changed(_):
                comp["exponents"][param_key] = current_exp()
                self._param_changed(comp, param_key, spin.value(), current_exp())

            slider.valueChanged.connect(update_spin_from_slider)
            spin.valueChanged.connect(update_slider_from_spin)
            min_box.valueChanged.connect(update_ranges)
            max_box.valueChanged.connect(update_ranges)
            exp_box.valueChanged.connect(on_exponent_changed)

            # initial values
            exp_box.setValue(exp_init)
            min_box.setValue(mantissa_min)
            max_box.setValue(mantissa_max)
            spin.setRange(min_box.value(), max_box.value())
            spin.setValue(mantissa_init)
            update_slider_from_spin(spin.value())

            return slider, spin, min_box, max_box, exp_box

        param_widgets = {}
        for name in self.param_names:
            v_init = comp["mantissas"][name]
            v_min, v_max = comp["ranges"][name]
            exp_init = comp["exponents"].get(name, 0)
            s, sp, min_box, max_box, exp_box = make_row(name, name, v_init, v_min, v_max, exp_init)
            param_widgets[name] = {
                "slider": s,
                "spin": sp,
                "min": min_box,
                "max": max_box,
                "exp": exp_box,
            }

        def on_visible_changed(state):
            comp["visible"] = state == Qt.CheckState.Checked
            self.update_curves()

        def on_delete():
            self._delete_component(comp)

        def on_label_changed(text):
            comp["label"] = text.strip() or comp.get("label") or group.title()
            group.setTitle(comp["label"])
            self.update_curves()

        # label edit
        lbl_edit = QLineEdit(comp.get("label", group.title()))
        lbl_edit.setPlaceholderText("Component name")
        lbl_edit.setFixedWidth(180)
        lbl_edit.editingFinished.connect(lambda: on_label_changed(lbl_edit.text()))
        header_layout.insertWidget(0, lbl_edit)

        # color edit
        color_edit = QLineEdit(comp.get("color", ""))
        color_edit.setPlaceholderText("#RRGGBB or name")
        color_edit.setFixedWidth(110)

        def on_color_changed():
            text = color_edit.text().strip()
            comp["color"] = text if text else comp.get("color", "")
            self._update_component_pen(comp)
            self.update_curves()

        color_edit.editingFinished.connect(on_color_changed)
        header_layout.insertWidget(1, color_edit)

        chk_visible.stateChanged.connect(on_visible_changed)
        btn_delete.clicked.connect(on_delete)

        comp["widgets"].update({
            "group": group,
            "visible": chk_visible,
            "delete": btn_delete,
            "color": color_edit,
            "label_edit": lbl_edit,
            "params": param_widgets,
        })

        comp["curve"] = self.plot(self.x_hr, np.zeros_like(self.x_hr), pen=self._make_component_pen(comp), name=None)
        self.comp_layout.addWidget(group)

    # --------------------------------------------------------
    # Component removal
    # --------------------------------------------------------
    def _clear_components(self):
        for c in list(self.components):
            if c.get("curve") is not None:
                self.plot_widget.removeItem(c["curve"])
            if "widgets" in c and c["widgets"].get("group") is not None:
                c["widgets"]["group"].setParent(None)
        self.components.clear()

    def _delete_component(self, comp):
        if comp.get("curve") is not None:
            self.plot_widget.removeItem(comp["curve"])
        self.components.remove(comp)
        w = comp["widgets"].get("group")
        if w is not None:
            w.setParent(None)
        self.update_curves()

    # --------------------------------------------------------
    # Parameter change handler
    # --------------------------------------------------------
    def _param_changed(self, comp, key, mantissa, exponent=None):
        # mantissa is what UI controls; exponent optionally passed
        if exponent is not None:
            comp["exponents"][key] = int(exponent)
        exp = comp["exponents"].get(key, 0)
        comp.setdefault("mantissas", {})
        comp["mantissas"][key] = float(mantissa)
        comp["params"][key] = float(self.combine_mantissa_exponent(mantissa, exp))
        self.update_curves()

    def _make_component_pen(self, comp):
        color = self.parse_color(comp.get("color", "#000000"), "#000000")
        alpha = 200 if comp["type"] == "pos" else 220
        style = Qt.PenStyle.SolidLine if comp["type"] == "pos" else Qt.PenStyle.DashLine
        return pg.mkPen(color=color, width=1.5, style=style)

    def _update_component_pen(self, comp):
        if comp.get("curve") is None:
            return
        comp["curve"].setPen(self._make_component_pen(comp))

    def update_x_grid(self):
        try:
            x_min = self.x_range_min()
            x_max = self.x_range_max()
        except Exception:
            return
        if x_max <= x_min:
            return
        N_high = max(2000, len(self.x_data) * 5)
        self.x_hr = np.linspace(x_min, x_max, N_high)
        self.range_dirty = True
        self.update_curves()

    def update_view_ranges_only(self):
        self.apply_view_ranges()

    # --------------------------------------------------------
    # Evaluate model and update curves
    # --------------------------------------------------------
    def evaluate_function(self, x, params):
        local_ns = dict(self.safe_namespace)
        local_ns.update(params)
        local_ns["x"] = x
        result = eval(self.function_expr, {"__builtins__": {}}, local_ns)
        arr = np.asarray(result, dtype=float)

        # Allow constant (scalar) functions by broadcasting across x
        if arr.shape == () or arr.size == 1:
            return np.full_like(x, float(arr))

        if arr.shape[0] != len(x):
            raise ValueError(f"Function output length {arr.shape[0]} does not match x length {len(x)}.")

        return arr

    def _info_label_prefix(self):
        return f"CSV format: {'value/error pairs' if self.has_error_columns else 'values only'}"

    def function_expr_to_latex(self, expr: str):
        # Minimal LaTeX-ish rendering; pyqtgraph legend does not render LaTeX,
        # but this keeps the string readable.
        latex = expr.replace("np.", "")
        latex = latex.replace("**", "^")
        latex = latex.replace("*", r" \cdot ")

        # crude sqrt conversion
        def _sqrt_replace(s):
            import re
            pattern = re.compile(r"sqrt\\(([^()]+)\\)")
            while True:
                new_s = pattern.sub(r"\\sqrt{\1}", s)
                if new_s == s:
                    break
                s = new_s
            return s

        latex = _sqrt_replace(latex)
        return latex

    def update_formula_legend(self):
        if self.legend is None:
            return
        if self.formula_item is not None:
            try:
                self.legend.removeItem(self.formula_item)
            except Exception:
                pass

        latex_expr = self.function_expr_to_latex(self.function_expr)
        self.formula_item = pg.PlotDataItem([], [], pen=pg.mkPen(color=(0, 0, 0, 0)))
        self.legend.addItem(self.formula_item, f"$f(x) = {latex_expr}$")

    def update_curves(self):
        info_prefix = self._info_label_prefix()

        if not self.components:
            self.fit_curve.setVisible(False)
            self.sum_pos_curve.setVisible(False)
            self.sum_neg_curve.setVisible(False)
            self.info_label.setText(f"{info_prefix}\nNo components yet.")
            self.apply_view_ranges()
            self.update_legend(None, None, None)
            return

        total = np.zeros_like(self.x_hr, dtype=float)
        y_pos_hr = np.zeros_like(self.x_hr, dtype=float)
        y_neg_hr = np.zeros_like(self.x_hr, dtype=float)

        info_lines = []
        for i, c in enumerate(self.components):
            try:
                y_single = np.array(self.evaluate_function(self.x_hr, c["params"]), dtype=float)
            except Exception as exc:
                self.info_label.setText(f"Error evaluating function: {exc}")
                return

            if c["type"] == "pos":
                total += y_single
                y_pos_hr += y_single
            else:
                total -= y_single
                y_neg_hr += y_single

            curve = c.get("curve")
            if curve is not None:
                if c["visible"]:
                    curve.setData(self.x_hr, y_single)
                    curve.setVisible(True)
                else:
                    curve.setVisible(False)

            param_str = ", ".join(f"{p}={c['params'][p]:.3g}" for p in self.param_names)
            info_lines.append(
                f"{i}: {'+' if c['type']=='pos' else '-'}  {param_str}, vis={int(c['visible'])}"
            )

        info_text = "\n".join(info_lines)
        if info_prefix:
            info_text = f"{info_prefix}\n{info_text}"
        self.info_label.setText(info_text)

        if self.show_total_curve:
            self.fit_curve.setData(self.x_hr, total)
            self.fit_curve.setVisible(True)
        else:
            self.fit_curve.setData([], [])
            self.fit_curve.setVisible(False)

        if self.show_sum_pos:
            self.sum_pos_curve.setData(self.x_hr, y_pos_hr)
            self.sum_pos_curve.setVisible(True)
        else:
            self.sum_pos_curve.setData([], [])
            self.sum_pos_curve.setVisible(False)

        if self.show_sum_neg:
            self.sum_neg_curve.setData(self.x_hr, np.abs(y_neg_hr))
            self.sum_neg_curve.setVisible(True)
        else:
            self.sum_neg_curve.setData([], [])
            self.sum_neg_curve.setVisible(False)

        self.apply_view_ranges()
        self.update_legend(y_pos_hr, y_neg_hr, total)

    # --------------------------------------------------------
    # Save cleaned data (negative noise removed)
    # --------------------------------------------------------
    def save_cleaned_negative(self):
        if not self.components:
            print("No components to clean with.")
            return

        y_neg = np.zeros_like(self.x_data, dtype=float)

        for c in self.components:
            if c["type"] != "neg":
                continue
            try:
                y_component = np.array(self.evaluate_function(self.x_data, c["params"]), dtype=float)
            except Exception as exc:
                print(f"Error evaluating function for negative component: {exc}")
                return
            y_neg += y_component

        if np.allclose(y_neg, 0):
            print("No negative components found. Nothing to clean.")
            return

        y_clean = self.y_data - y_neg

        out_df = pd.DataFrame({
            "x": self.x_data,
            "y_clean": y_clean
        })

        out_path = self.input_path.with_name(self.input_path.stem + "_cleaned.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved cleaned data to: {out_path}")

    # --------------------------------------------------------
    # Export LaTeX table
    # --------------------------------------------------------
    def export_latex_table(self):
        if not self.components:
            print("No components to export.")
            return

        columns = ["$i$", "sign"] + [f"${name}$" for name in self.param_names]
        lines = []
        lines.append(r"\begin{tabular}{%s}" % (" ".join(["c"] * len(columns))))
        lines.append(r"\hline")
        lines.append(" & ".join(columns) + r" \\")
        lines.append(r"\hline")

        for i, c in enumerate(self.components):
            sign = "+" if c["type"] == "pos" else "-"
            params_text = " & ".join(f"{c['params'][p]:.3g}" for p in self.param_names)
            line = f"{i} & {sign}"
            if params_text:
                line += f" & {params_text}"
            line += r" \\"
            lines.append(line)

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        latex = "\n".join(lines)

        latex_formula = self.function_expr_to_latex(self.function_expr)
        latex += "\n\n" + r"$f(x) = %s$" % latex_formula

        out_path = self.input_path.with_name("params_table.tex")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex)

        print("\n--- LaTeX table (also written to params_table.tex) ---\n")
        print(latex)
        print("\n------------------------------------------------------\n")

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------
    def save_settings(self):
        import json

        slot = (self.settings_slot_edit.text().strip() or "fit1").replace(" ", "_")

        comps = []
        for c in self.components:
            comps.append({
                "type": c["type"],
                "visible": c["visible"],
                "params": c["params"],
                "ranges": c["ranges"],
                "exponents": c.get("exponents", {}),
                "mantissas": c.get("mantissas", {}),
                "label": c.get("label"),
                "color": c.get("color"),
            })

        data = {
            "function_expr": self.function_expr,
            "param_names": self.param_names,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "components": comps,
            "param_defaults": self.param_default_settings,
            "x_range": self.x_range_settings,
            "y_range": self.y_range_settings,
        }

        out_path = self._settings_path(slot)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Settings saved to slot '{slot}' at {out_path}")

    def load_settings(self):
        import json

        slot = (self.settings_slot_edit.text().strip() or "fit1").replace(" ", "_")
        path = self._settings_path(slot)
        if not path.exists():
            print("No settings file found:", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expr = data.get("function_expr", self.function_expr)
        try:
            self.param_names = self.extract_parameter_names(expr)
            self.function_expr = expr
        except SyntaxError as exc:
            print("Invalid function in settings:", exc)
            return

        self.safe_namespace = self.build_safe_namespace()
        self.function_edit.setText(self.function_expr)
        self.param_default_settings = data.get("param_defaults", {}) or {}
        # ensure defaults include all params
        self.ensure_default_settings()
        self.x_range_settings = data.get("x_range", self.x_range_settings) or self.x_range_settings
        self.y_range_settings = data.get("y_range", self.y_range_settings) or self.y_range_settings
        if not self.y_range_settings:
            self.reset_y_range_defaults()
        self.range_dirty = True

        self.x_label = data.get("x_label", self.x_label)
        self.y_label = data.get("y_label", self.y_label)
        self.x_label_edit.setText(self.x_label)
        self.y_label_edit.setText(self.y_label)
        self.apply_axis_labels()

        self.build_default_controls()
        self._clear_components()

        for entry in data.get("components", []):
            params = entry.get("params", {})
            ranges = entry.get("ranges", {})
            exponents = entry.get("exponents", {})
            mantissas = entry.get("mantissas", {})
            params_with_label = dict(params)
            if "label" in entry:
                params_with_label["label"] = entry["label"]
            comp = self._add_component(entry.get("type", "pos"), params_with_label, ranges, exponents, mantissas, visible=entry.get("visible", True), update=False, color=entry.get("color"))

            # restore visibility checkbox
            comp["widgets"]["visible"].setChecked(comp["visible"])
            comp["widgets"]["label_edit"].setText(comp.get("label", comp["widgets"]["group"].title()))
            comp["widgets"]["group"].setTitle(comp.get("label", comp["widgets"]["group"].title()))
            comp["widgets"]["color"].setText(comp.get("color", ""))
            self._update_component_pen(comp)

            for name in self.param_names:
                controls = comp["widgets"]["params"][name]
                exp_val = comp["exponents"].get(name, 0)
                v_min, v_max = comp["ranges"][name]
                val = comp["mantissas"][name]
                controls["exp"].setValue(exp_val)
                controls["min"].setValue(v_min)
                controls["max"].setValue(v_max)
                controls["spin"].setRange(controls["min"].value(), controls["max"].value())
                controls["spin"].setValue(val)

        self.update_formula_legend()
        self.update_curves()
        print(f"Settings loaded from slot '{slot}' ({path}).")

    # --------------------------------------------------------
    # Function and axis updates
    # --------------------------------------------------------
    def apply_function_expression(self):
        expr = self.function_edit.text().strip()
        if not expr:
            return
        try:
            new_params = self.extract_parameter_names(expr)
        except SyntaxError as exc:
            self.info_label.setText(f"Invalid function: {exc}")
            return

        self.function_expr = expr
        self.param_names = new_params
        self.safe_namespace = self.build_safe_namespace()
        self.reset_param_defaults()
        self.build_default_controls()
        self.rebuild_components_for_new_params()
        self.update_formula_legend()
        self.update_curves()

    def rebuild_components_for_new_params(self):
        if not self.components:
            self._clear_components()
            self.reset_param_defaults()
            self.build_default_controls()
            self._add_component("pos", update=False)
            return

        old_components = list(self.components)
        self._clear_components()
        self.reset_param_defaults()
        self.build_default_controls()

        for old in old_components:
            defaults, default_ranges, default_exponents = self.default_param_state()
            params = {name: old["params"].get(name, defaults[name]) for name in self.param_names}
            ranges = {name: old["ranges"].get(name, default_ranges[name]) for name in self.param_names}
            exps = {name: old.get("exponents", {}).get(name, default_exponents.get(name, 0)) for name in self.param_names}
            mantissas = {}
            for name in self.param_names:
                factor = 10 ** exps.get(name, 0) if exps.get(name, 0) else 1
                mantissas[name] = params[name] / factor if factor != 0 else params[name]
            comp = self._add_component(old["type"], params, ranges, exps, mantissas, visible=old["visible"], update=False, color=old.get("color"))
            comp["widgets"]["visible"].setChecked(comp["visible"])
            comp["widgets"]["label_edit"].setText(old.get("label", comp["widgets"]["group"].title()))
            comp["widgets"]["group"].setTitle(old.get("label", comp["widgets"]["group"].title()))
            comp["widgets"]["color"].setText(old.get("color", ""))
            self._update_component_pen(comp)
            for name in self.param_names:
                controls = comp["widgets"]["params"][name]
                exp_val = comp["exponents"].get(name, 0)
                v_min, v_max = comp["ranges"][name]
                controls["exp"].setValue(exp_val)
                controls["min"].setValue(v_min)
                controls["max"].setValue(v_max)
                controls["spin"].setRange(controls["min"].value(), controls["max"].value())
                controls["spin"].setValue(comp["mantissas"][name])

    def update_axis_labels_from_inputs(self):
        self.x_label = self.x_label_edit.text().strip() or self.x_label
        self.y_label = self.y_label_edit.text().strip() or self.y_label
        self.apply_axis_labels()

    def apply_axis_labels(self):
        self.plot_widget.setLabel("bottom", self.x_label)
        self.plot_widget.setLabel("left", self.y_label)

    def closeEvent(self, event):
        if hasattr(self, "control_window") and self.control_window is not None:
            self.control_window.close()
        super().closeEvent(event)

    def toggle_sum_curves(self, state):
        self.show_sum_curves = state == Qt.CheckState.Checked
        self.update_curves()
    def toggle_total_curve(self, state):
        self.show_total_curve = state == Qt.CheckState.Checked
        self.update_curves()

    def toggle_sum_pos_curve(self, state):
        self.show_sum_pos = state == Qt.CheckState.Checked
        self.update_curves()

    def toggle_sum_neg_curve(self, state):
        self.show_sum_neg = state == Qt.CheckState.Checked
        self.update_curves()

    def _settings_path(self, slot: str):
        if not slot:
            slot = "fit1"
        filename = f"{self.input_path.stem}_{slot}.settings.json"
        return self.input_path.with_name(filename)

    def apply_view_ranges(self):
        if not self.range_dirty:
            self._log_range_debug("apply_view_ranges skipped (clean)", None, None, None, None, None, None)
            return
        try:
            x_min = self.x_range_min()
            x_max = self.x_range_max()
            y_min = self.y_range_min()
            y_max = self.y_range_max()
        except Exception:
            return

        if x_max <= x_min or y_max <= y_min:
            return

        # No extra padding here; defaults already include 5% padding.
        x_pad = 0.0
        y_pad = 0.0

        view = self.plot_widget.getViewBox()
        view.disableAutoRange()
        view.setXRange(x_min - x_pad, x_max + x_pad, padding=0)
        view.setYRange(y_min - y_pad, y_max + y_pad, padding=0)
        self.range_dirty = False
        self._log_range_debug("apply_view_ranges", x_min, x_max, y_min, y_max, x_pad, y_pad)
        self._log_view_state(view)

    def save_plot_png(self):
        # ensure ranges applied before export
        self.apply_view_ranges()
        exporter = ImageExporter(self.plot_widget.plotItem)
        slot = (self.settings_slot_edit.text().strip() or "fit1").replace(" ", "_")
        default_path = self.input_path.with_name(f"{self.input_path.stem}_{slot}.png")
        try:
            exporter.export(str(default_path))
            print(f"Plot saved to {default_path}")
        except Exception as exc:
            print(f"Failed to save plot: {exc}")

    def _legend_label_for_component(self, comp):
        param_str = ", ".join(f"{p}={comp['params'][p]:.3g}" for p in self.param_names)
        return f"{comp.get('label', '')} ({param_str})" if param_str else comp.get("label", "")

    def _log_range_debug(self, label, x_min, x_max, y_min, y_max, x_pad, y_pad):
        # Allow None when skipped
        if None in (x_min, x_max, y_min, y_max, x_pad, y_pad):
            print(f"[debug] {label}: skipped or not ready")
            return
        print(f"[debug] {label}: "
              f"x_min={x_min:.6g}, x_max={x_max:.6g}, x_pad={x_pad:.6g}; "
              f"y_min={y_min:.6g}, y_max={y_max:.6g}, y_pad={y_pad:.6g}; "
              f"x_range=({x_min - x_pad:.6g}, {x_max + x_pad:.6g}), "
              f"y_range=({y_min - y_pad:.6g}, {y_max + y_pad:.6g})")

    def _log_view_state(self, view):
        try:
            (x0, x1), (y0, y1) = view.viewRange()
            print(f"[debug] viewRange: x=({x0:.6g},{x1:.6g}) y=({y0:.6g},{y1:.6g})")
        except Exception:
            pass

    def update_legend(self, y_pos_hr, y_neg_hr, total):
        try:
            self.legend.clear()
        except Exception:
            # fallback: remove items manually
            if hasattr(self.legend, "items"):
                for sample, _ in list(self.legend.items):
                    try:
                        self.legend.removeItem(sample.item)
                    except Exception:
                        pass

        if self.data_curve.isVisible():
            self.legend.addItem(self.data_curve, "Data")

        if self.show_total_curve and self.fit_curve.isVisible():
            self.legend.addItem(self.fit_curve, "Total")

        if self.show_sum_pos and self.sum_pos_curve.isVisible():
            self.legend.addItem(self.sum_pos_curve, "Sum +")

        if self.show_sum_neg and self.sum_neg_curve.isVisible():
            self.legend.addItem(self.sum_neg_curve, "Sum -")

        for i, comp in enumerate(self.components):
            if not comp.get("visible", True):
                continue
            label = self._legend_label_for_component(comp)
            self.legend.addItem(comp["curve"], label or f"Comp {i}")

        # formula entry (dummy item)
        formula_item = pg.PlotDataItem([], [])
        self.legend.addItem(formula_item, f"f(x) = {self.function_expr_to_latex(self.function_expr)}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive function fitter and plotter.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="physics-experiment-eval-tool/fitter/function_fitter_test_data/Pb_energy.csv",
        help="Path to the CSV file containing the measurements.",
    )
    parser.add_argument(
        "--has-error-columns",
        action="store_true",
        help="Treat the CSV as value/error pairs (val, err_val, val2, err_val2, ...).",
    )
    return parser.parse_args()


def load_input_data(input_path: Path, has_error_columns: bool):
    df = pd.read_csv(input_path)
    if df.isnull().any().any():
        raise ValueError("CSV contains missing values; every row must provide the expected number of entries.")
    num_cols = df.shape[1]

    if has_error_columns:
        if num_cols < 4 or num_cols % 2 != 0:
            raise ValueError(
                f"Expected value/error pairs (val, err_val, val2, err_val2, ...) but found {num_cols} columns."
            )
        x = df.iloc[:, 0].values.astype(float)
        x_err = df.iloc[:, 1].values.astype(float)
        y = df.iloc[:, 2].values.astype(float)
        y_err = df.iloc[:, 3].values.astype(float)
    else:
        if num_cols != 2:
            raise ValueError(
                f"Expected exactly two columns (x, y) when error columns are disabled, but found {num_cols}."
                " If your CSV stores value/error pairs, rerun with --has-error-columns."
            )
        x = df.iloc[:, 0].values.astype(float)
        y = df.iloc[:, 1].values.astype(float)
        x_err = None
        y_err = None

    return x, y, x_err, y_err


def main():
    args = parse_args()
    input_path = Path(args.csv_path)

    if not input_path.exists():
        print(f"CSV not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        x, y, x_err, y_err = load_input_data(input_path, args.has_error_columns)
    except Exception as exc:
        print(f"Error while reading {input_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    win = GaussianFitUI(x, y, input_path, x_err=x_err, y_err=y_err, has_error_columns=args.has_error_columns)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
