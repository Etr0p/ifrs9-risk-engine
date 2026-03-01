import pandas as pd
import numpy as np
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.engine.staging import compute_sicr_score, StagingEngine
from ifrs9_cockpit.config import PREDEFINED_SCENARIOS, SCENARIO_BASE, SICR_CONFIG

def diag_sicr():
    print("Chargement des données...")
    df_credit, _, _ = generate_dataset(n_clients=5000)
    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_curr_base = pd_suite.predict(df_credit)["LR_WoE"]
    pd_orig = df_credit["pd_origination"].values
    dpd = df_credit["dpd"].values
    
    scenarios = ["Central", "Crise financiere (GFC)", "Stagflation", "Reprise"]
    
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel
    from ifrs9_cockpit.engine.staging import compute_macro_z

    ecl_calc = ECLCalculator(LGDModel(), EADModel())

    for name in scenarios:
        p = PREDEFINED_SCENARIOS[name]
        un = float(SCENARIO_BASE.unemployment_rate + abs(p["unemployment_bipolar"]))
        ir = float(SCENARIO_BASE.interest_rate + p["interest_rate_bp"] / 100.0)
        gdp, hpi, infl = float(p["gdp_pct"]), float(p["hpi_pct"]), float(p["inflation_pct"])
        
        macro_params = {
            "unemployment_rate": un,
            "gdp_growth": gdp,
            "interest_rate": ir,
            "hpi_growth": hpi,
            "inflation_rate": infl
        }
        
        pd_stressed = ecl_calc._adjust_pd_for_scenario(
            pd_curr_base, df_credit, SCENARIO_BASE,
            unemployment_override=un, gdp_override=gdp, interest_rate_override=ir,
            hpi_override=hpi, inflation_override=infl
        )
        
        scores = compute_sicr_score(pd_stressed, pd_orig, dpd, macro_params)
        
        pd_ratio = np.clip(pd_stressed / np.maximum(pd_orig, 1e-6) - 1, 0.0, 5.0)
        pd_delta = np.maximum(0, pd_stressed - pd_orig)
        mz = compute_macro_z(macro_params)
        
        print(f"\nScenario: {name}")
        print(f"  Score mean: {scores.mean():.3f}, std: {scores.std():.3f}")
        print(f"  Score max: {scores.max():.3f}, min: {scores.min():.3f}")
        print(f"  Stage 2 trigger (score > {SICR_CONFIG.threshold}): {(scores > SICR_CONFIG.threshold).mean():.1%}")
        print(f"  Avg pd_ratio (weighted): {(pd_ratio * SICR_CONFIG.w_pd_ratio).mean():.3f}")
        print(f"  Avg pd_delta (weighted): {(pd_delta * SICR_CONFIG.w_pd_delta).mean():.3f}")
        print(f"  Macro Z (weighted): {(mz * SICR_CONFIG.w_macro):.3f}")
        
if __name__ == "__main__":
    diag_sicr()
