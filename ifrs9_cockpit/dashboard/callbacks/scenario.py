"""Callbacks scenario -- presets, sliders, incoherence."""
from __future__ import annotations

from dash import Input, Output, no_update
import dash_bootstrap_components as dbc

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.config import PREDEFINED_SCENARIOS, MACRO_INCOHERENCE_RULES, SCENARIO_BASE


def register_scenario(app):
    # ══════════════════════════════════════════════
    # CALLBACK 0b : RST toggle (enable/disable target input)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.RST_TARGET, "disabled"),
        Input(ids.RST_ENABLED, "value"),
    )
    def toggle_rst_target(rst_value):
        """Active/desactive le champ cible ECL selon la checkbox RST."""
        return not (rst_value and "on" in rst_value)

    # ══════════════════════════════════════════════
    # CALLBACK 1 : Scenario Preset
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.SL_INTEREST_RATE, "value"),
         Output(ids.SL_UNEMPLOYMENT, "value"),
         Output(ids.SL_GDP, "value"),
         Output(ids.SL_HPI, "value"),
         Output(ids.SL_INFLATION, "value")],
        Input(ids.SCENARIO_SELECTOR, "value"),
        prevent_initial_call=True,
    )
    def apply_scenario(scenario_name):
        """Pre-remplit les 5 sliders depuis un scenario predefined."""
        if scenario_name == "Manuel" or scenario_name not in PREDEFINED_SCENARIOS:
            return no_update, no_update, no_update, no_update, no_update
        preset = PREDEFINED_SCENARIOS[scenario_name]
        return (
            preset["interest_rate_bp"],
            preset["unemployment_bipolar"],
            preset["gdp_pct"],
            preset["hpi_pct"],
            preset["inflation_pct"],
        )

    # ══════════════════════════════════════════════
    # CALLBACK 1b : Slider value display (live update)
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.VAL_INTEREST_RATE, "children"),
         Output(ids.VAL_UNEMPLOYMENT, "children"),
         Output(ids.VAL_GDP, "children"),
         Output(ids.VAL_HPI, "children"),
         Output(ids.VAL_INFLATION, "children")],
        [Input(ids.SL_INTEREST_RATE, "value"),
         Input(ids.SL_UNEMPLOYMENT, "value"),
         Input(ids.SL_GDP, "value"),
         Input(ids.SL_HPI, "value"),
         Input(ids.SL_INFLATION, "value")],
    )
    def update_slider_values(ir, unemp, gdp, hpi, infl):
        """Met a jour l'affichage de la valeur courante des sliders."""
        return (
            f"{ir}" if ir is not None else "0",
            f"{unemp}" if unemp is not None else "0",
            f"{gdp}" if gdp is not None else "0",
            f"{hpi}" if hpi is not None else "0",
            f"{infl}" if infl is not None else "0",
        )

    # ══════════════════════════════════════════════
    # CALLBACK 2 : Incoherence Detection
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.INCOHERENCE_ALERTS, "children"),
        [Input(ids.SL_INTEREST_RATE, "value"),
         Input(ids.SL_UNEMPLOYMENT, "value"),
         Input(ids.SL_GDP, "value"),
         Input(ids.SL_HPI, "value"),
         Input(ids.SL_INFLATION, "value")],
    )
    def check_incoherence(ir_bp, unemp, gdp, hpi, infl):
        """Evalue les regles d'incoherence macro et affiche les alertes."""
        slider_vals = {
            "interest_rate_bp": ir_bp or 0.0,
            "unemployment_bipolar": unemp or 0.0,
            "gdp_pct": gdp if gdp is not None else SCENARIO_BASE.gdp_growth,
            "hpi_pct": hpi if hpi is not None else SCENARIO_BASE.hpi_growth,
            "inflation_pct": infl if infl is not None else SCENARIO_BASE.inflation_rate,
        }
        alerts = []
        for rule in MACRO_INCOHERENCE_RULES:
            all_met = True
            for var, op, threshold in rule.conditions:
                val = slider_vals.get(var, 0.0)
                if op == "gt" and not (val > threshold):
                    all_met = False
                elif op == "lt" and not (val < threshold):
                    all_met = False
            if all_met:
                alerts.append(
                    dbc.Alert(
                        f"Incoherence : {rule.description}",
                        color="warning",
                        className="mb-2",
                    )
                )
        return alerts if alerts else []
