
from ifrs9_cockpit.data.generator import generate_dataset
import numpy as np

df_c, _, _ = generate_dataset(n_clients=5000)
print(f"pd_orig mean: {df_c['pd_origination'].mean():.4f}")
print(f"pd_orig std: {df_c['pd_origination'].std():.4f}")
print(f"default_flag mean: {df_c['default_flag'].mean():.4f}")
# Check if pd_orig is same as pd_latent if available
if 'pd_latent' in df_c.columns:
    print(f"Mean diff (orig - latent): {(df_c['pd_origination'] - df_c['pd_latent']).mean():.6f}")
