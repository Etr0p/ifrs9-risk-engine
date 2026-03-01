
import pandas as pd
import numpy as np
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.config import SCENARIO_BASE

def diag():
    df_credit, df_pe, _ = generate_dataset(n_clients=1000)
    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_curr = pd_suite.predict(df_credit)["LR_WoE"]
    pd_orig = df_credit["pd_origination"].values
    
    ecl_calc = ECLCalculator(LGDModel(), EADModel())
    
    # 1. Base call (no overrides)
    res_base = ecl_calc.calculate(df_credit, pd_curr, pd_orig)
    ecl_base = res_base["ecl_weighted"].sum()
    
    # 2. Stress call (neutral overrides)
    res_stress_neutral = ecl_calc.calculate(
        df_credit, pd_curr, pd_orig,
        unemployment_override=SCENARIO_BASE.unemployment_rate,
        gdp_override=SCENARIO_BASE.gdp_growth,
        interest_rate_override=SCENARIO_BASE.interest_rate,
        hpi_override=SCENARIO_BASE.hpi_growth,
        inflation_override=SCENARIO_BASE.inflation_rate
    )
    ecl_stress_neutral = res_stress_neutral["ecl_weighted"].sum()
    
    print(f"ECL Base: {ecl_base:.2f}")
    print(f"ECL Stress Neutral: {ecl_stress_neutral:.2f}")
    print(f"Delta: {(ecl_stress_neutral - ecl_base) / ecl_base:.2%}")
    
    # Check if staging is identical
    diff_stages = (res_base["stage"] != res_stress_neutral["stage"]).sum()
    print(f"Staging diffs: {diff_stages}")

if __name__ == "__main__":
    diag()
