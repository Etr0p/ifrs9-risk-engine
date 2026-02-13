"""Constantes d'IDs pour les composants Dash -- IFRS 9 Risk Cockpit.

Chaque composant Dash interactif (Input, Output, State) est reference
par un ID unique. Centraliser ces IDs ici evite les erreurs de typo
dans les callbacks et facilite le refactoring.

Convention de nommage :
    - kebab-case (standard HTML/Dash)
    - prefixe par section fonctionnelle (sl- pour sliders, btn- pour boutons, etc.)
    - suffixe -selector, -output, -content pour clarifier le role
"""

from __future__ import annotations

# ──────────────────────────────────────────────
# SIDEBAR INPUTS
# ──────────────────────────────────────────────
SCENARIO_SELECTOR = "scenario-selector"
SL_INTEREST_RATE = "sl-interest-rate"
SL_UNEMPLOYMENT = "sl-unemployment"
SL_GDP = "sl-gdp"
SL_HPI = "sl-hpi"
SL_INFLATION = "sl-inflation"
PD_MODEL_SELECTOR = "pd-model-selector"
PE_ALLOCATION = "pe-allocation"
RW_PE_SELECTOR = "rw-pe-selector"
RST_ENABLED = "rst-enabled"
RST_TARGET = "rst-target"

# Slider value display (inline text next to label)
VAL_INTEREST_RATE = "val-interest-rate"
VAL_UNEMPLOYMENT = "val-unemployment"
VAL_GDP = "val-gdp"
VAL_HPI = "val-hpi"
VAL_INFLATION = "val-inflation"
VAL_PE_ALLOC = "val-pe-alloc"

# ──────────────────────────────────────────────
# STORES (server-client data transfer via dcc.Store)
# ──────────────────────────────────────────────
STORE_PIPELINE = "store-pipeline"

# ──────────────────────────────────────────────
# MAIN AREA — DYNAMIC CONTENT
# ──────────────────────────────────────────────
KPI_ROW = "kpi-row"
CLASSIFICATION_ROW = "classification-row"
AI_NARRATIVE = "ai-narrative"
SCENARIO_BANNER = "scenario-banner"
INCOHERENCE_ALERTS = "incoherence-alerts"

# ──────────────────────────────────────────────
# SCORE CARDS (clickable summary cards)
# ──────────────────────────────────────────────
PE_CARD = "pe-card"
CREDIT_CARD = "credit-card"

# ──────────────────────────────────────────────
# MODALS (detail overlays)
# ──────────────────────────────────────────────
MODAL_PE = "modal-pe"
MODAL_CREDIT = "modal-credit"
MODAL_PE_BODY = "modal-pe-body"
MODAL_CREDIT_BODY = "modal-credit-body"

# ──────────────────────────────────────────────
# ARBITRAGE SECTION (drill-down sectoriel)
# ──────────────────────────────────────────────
ARBITRAGE_SECTION = "arbitrage-section"
RST_RESULTS = "rst-results"
DRILL_SECTOR_SELECTOR = "drill-sector-selector"
DRILL_SECTOR_OUTPUT = "drill-sector-output"

# ──────────────────────────────────────────────
# COLLAPSIBLE SECTIONS (progressive disclosure)
# ──────────────────────────────────────────────
BTN_PERF = "btn-perf"
BTN_SHAP = "btn-shap"
BTN_EXPORT = "btn-export"
COLLAPSE_PERF = "collapse-perf"
COLLAPSE_SHAP = "collapse-shap"
COLLAPSE_EXPORT = "collapse-export"
PERF_CONTENT = "perf-content"
SHAP_CONTENT = "shap-content"
EXPORT_CONTENT = "export-content"

# ──────────────────────────────────────────────
# SHAP INDIVIDUAL EXPLAINABILITY
# ──────────────────────────────────────────────
SHAP_CLIENT_SELECTOR = "shap-client-selector"
SHAP_INDIVIDUAL_OUTPUT = "shap-individual-output"

# ──────────────────────────────────────────────
# EXPORT DOWNLOADS
# ──────────────────────────────────────────────
DL_EXCEL = "dl-excel"
DL_CRO_TXT = "dl-cro-txt"
DL_AI_TXT = "dl-ai-txt"
DL_LATEX = "dl-latex"
BTN_DL_EXCEL = "btn-dl-excel"
BTN_DL_CRO = "btn-dl-cro"
BTN_DL_AI = "btn-dl-ai"
BTN_DL_LATEX = "btn-dl-latex"

# ──────────────────────────────────────────────
# LOADING OVERLAY
# ──────────────────────────────────────────────
LOADING_PIPELINE = "loading-pipeline"
