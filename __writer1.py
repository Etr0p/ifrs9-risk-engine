
import pathlib

script_lines = [
    'import sys, warnings',
    'sys.path.insert(0, ".")',
    'warnings.filterwarnings("ignore")',
    '',
    'import numpy as np',
    'import pandas as pd',
    'from ifrs9_cockpit.config import SCENARIO_BASE, PREDEFINED_SCENARIOS, BASEL_CONFIG, SECTORS',
    'from ifrs9_cockpit.data.generator import generate_dataset',
    'from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator',
    'from ifrs9_cockpit.models.pd_model import PDModelSuite',
    'from ifrs9_cockpit.models.pe_model import PECalculator',
    'from ifrs9_cockpit.engine.comparator import PortfolioComparator',
    '',
    'def slider_to_macro(sliders):',
    '    return {',
    '        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(sliders.get("unemployment_bipolar", 0)),',
    '        "interest_rate": SCENARIO_BASE.interest_rate + sliders.get("interest_rate_bp", 0) / 100,',
    '        "gdp_growth": sliders.get("gdp_pct", SCENARIO_BASE.gdp_growth),',
    '        "hpi_growth": sliders.get("hpi_pct", SCENARIO_BASE.hpi_growth),',
    '        "inflation_rate": sliders.get("inflation_pct", SCENARIO_BASE.inflation_rate),',
    '    }',
    '',
    'def run_pipeline(macro, seed=123):',
    '    df_c, df_pe, _ = generate_dataset(n_clients=1000, seed=seed)',
    '    suite = PDModelSuite()',
    '    suite.fit(df_c.copy())',
    '    calc = ECLCalculator(df_c.copy(), suite, macro)',
    '    res = calc.compute()',
    '    pe_calc = PECalculator(df_pe.copy(), macro)',
    '    res_pe = pe_calc.compute()',
    '    for df in [res, res_pe]:',
    '        if "sector" in df.columns and "segment" not in df.columns:',
    '            df["segment"] = df["sector"]',
    '    return res, res_pe, df_c, df_pe',
]

pathlib.Path(r"C:	out\cours\programme\__audit_4sc.py").write_text(chr(10).join(script_lines), encoding="utf-8")
print("Part 1 written")
