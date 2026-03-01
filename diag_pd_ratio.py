import pandas as pd
import numpy as np
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.config import SCENARIO_BASE

def diag_pd_ratio():
    df_c, _, _ = generate_dataset(n_clients=5000)
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    pd_suite = PDModelSuite()
    pd_suite.fit(df_c)
    pd_curr_base = pd_suite.predict(df_c)["LR_WoE"]
    pd_orig = df_c['pd_origination'].values
    
    ratio = pd_curr_base / np.maximum(pd_orig, 1e-6) - 1
    ratio_capped = np.clip(ratio, 0, 5.0)
    
    print(f"PD Orig mean: {pd_orig.mean():.4f}, median: {np.median(pd_orig):.4f}")
    print(f"PD Curr Base mean: {pd_curr_base.mean():.4f}, median: {np.median(pd_curr_base):.4f}")
    print(f"Ratio mean: {ratio.mean():.4f}, median: {np.median(ratio):.4f}")
    print(f"Ratio capped mean: {ratio_capped.mean():.4f}")
    
    print("\nPD Orig distribution:")
    print(pd.Series(pd_orig).describe())
    
    print(f"\nPD Orig < 0.001: {(pd_orig < 0.001).sum()} / {len(pd_orig)}")
    if (pd_orig < 0.001).any():
        idx = np.where(pd_orig < 0.001)[0][0]
        print(f"Example: orig={pd_orig[idx]:.6f}, curr={pd_curr_base[idx]:.6f}, ratio={ratio[idx]:.4f}")

if __name__ == "__main__":
    diag_pd_ratio()
