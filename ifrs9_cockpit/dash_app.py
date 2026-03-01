"""Application Dash principale — IFRS 9 Risk Cockpit.

Point d'entree du dashboard interactif (alternative Dash a app.py Streamlit).
Orchestre le pipeline complet via callbacks reactifs :
    1. Sidebar : 5 sliders macro + config modele/PE/RST
    2. Pipeline 5 stages (ECL, PE, Comparaison, Optimisation, AI Analyst)
    3. Score cards PE / Credit avec modals de detail
    4. Arbitrage + sections pliables (Performance, SHAP, Export)

Lancer avec : python ifrs9_cockpit/dash_app.py
ou :          gunicorn ifrs9_cockpit.dash_app:server
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Permettre le lancement depuis n'importe quel repertoire
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dash
import dash_bootstrap_components as dbc

from ifrs9_cockpit.dashboard.cache import (
    load_data,
    train_pd_models,
    train_lgd_ead,
    force_models_cpu,
)
from ifrs9_cockpit.dashboard.layout import build_layout
from ifrs9_cockpit.dashboard import callbacks
# ──────────────────────────────────────────────
# INITIALISATION (donnees + modeles au demarrage)
# ──────────────────────────────────────────────
# Les fonctions load_data / train_pd_models / train_lgd_ead utilisent
# @lru_cache dans cache.py, donc le reload Werkzeug ne re-calcule rien.
print("[1/4] Chargement des donnees...")
_bundle = load_data()
df_credit, df_pe, df_history, df_balance_sheet = _bundle

print("[2/4] Chargement des modeles PD...")
pd_suite = train_pd_models()
force_models_cpu(pd_suite)

print("[3/4] Calibration LGD & EAD...")
lgd_model, ead_model = train_lgd_ead()

print("[4/4] Construction de l'application Dash...")

# ──────────────────────────────────────────────
# APPLICATION DASH
# ──────────────────────────────────────────────
# CSS and JS are auto-loaded from assets/ directory:
#   assets/custom-theme.css  — Steel Blue dark theme
#   assets/dark-dropdown.css — Dash 4.0 dropdown overrides
#   assets/zz-dark-override.js — MutationObserver for Dash design tokens
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="IFRS 9 Risk Cockpit",
    update_title="Calcul en cours...",
)
server = app.server  # Pour gunicorn / production WSGI

# Layout
app.layout = build_layout(list(pd_suite.results.keys()))

# Callbacks
callbacks.register(app, df_credit, df_pe, df_history, pd_suite, lgd_model, ead_model,
                   df_balance_sheet=df_balance_sheet, dataset_bundle=_bundle)

print("Application prete. Ouverture sur http://localhost:8050")

if __name__ == "__main__":
    app.run(
        debug=True,
        host="localhost",
        port=8050,
        # Hot reload CSS/JS/assets: le navigateur rafraichit automatiquement.
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=2.0,
        dev_tools_hot_reload_watch_interval=1.0,
        dev_tools_props_check=False,
        # Masquer la barre de debug en bas de page
        dev_tools_ui=False,
        # use_reloader=False: pas de relance du process Python.
        # Hot reload CSS/JS fonctionne toujours. Pour les .py, F5 dans le
        # navigateur suffit car les callbacks importent les modules a chaque appel.
        # Pour un changement de layout/callback, relancer le serveur manuellement.
        use_reloader=False,
    )
