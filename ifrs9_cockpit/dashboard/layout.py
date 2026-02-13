"""Arbre DOM complet (layout) du dashboard Dash IFRS 9 Risk Cockpit.

Ce module definit ``build_layout()`` qui retourne la hierarchie complete
des composants Dash constituant l'interface utilisateur.  La structure
suit le pattern *sidebar + main content* :

* **Sidebar** (fixe a gauche) : controles de stress test, configuration
  modele PD, allocation PE, et reverse stress test.
* **Main content** (droite) : header, banniere scenario, KPIs, cartes
  score (PE / Credit) avec modals de detail, sections collapsibles
  (Performance, SHAP, Export), et pied de page.

Les IDs de tous les composants interactifs proviennent de ``ids.py``
pour eviter les erreurs de typo dans les callbacks.

Usage::

    from ifrs9_cockpit.dashboard.layout import build_layout
    app.layout = build_layout(pd_model_names=["LR_WoE", "XGBoost", "TabNet"])
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard.components_dash import (
    build_header,
    build_sidebar,
    build_collapsible_button,
)


def build_layout(pd_model_names: list[str]) -> html.Div:
    """Build and return the complete Dash layout tree.

    The returned ``html.Div`` contains:

    1. A ``dcc.Store`` for pipeline data (memory-backed).
    2. The fixed sidebar with all interactive controls.
    3. The main content area with dynamic placeholders that are
       populated by callbacks at runtime.

    Args:
        pd_model_names: List of available PD model family names
            (e.g. ``["LR_WoE", "XGBoost", "TabNet"]``).  Passed
            through to ``build_sidebar()`` for the model selector
            dropdown.

    Returns:
        The root ``html.Div`` of the application.
    """
    return html.Div([
        # ── Data stores ──────────────────────────────────────────
        dcc.Store(id=ids.STORE_PIPELINE, storage_type="memory"),

        # ── Sidebar (fixed left) ────────────────────────────────
        build_sidebar(pd_model_names),

        # ── Main content area ───────────────────────────────────
        html.Div([
            # Header
            build_header(),

            # Dynamic content (updated by callbacks)
            html.Div(id=ids.INCOHERENCE_ALERTS),
            html.Div(id=ids.KPI_ROW),

            # RST results panel (visible only when RST is active)
            html.Div(id=ids.RST_RESULTS),

            # Score cards row (PE + Credit) — arbitrage-first layout
            dbc.Row([
                dbc.Col(html.Div(id=ids.PE_CARD, n_clicks=0), md=6),
                dbc.Col(html.Div(id=ids.CREDIT_CARD, n_clicks=0), md=6),
            ], className="my-4"),

            # Arbitrage section (directly after score cards)
            html.Div(id=ids.ARBITRAGE_SECTION, className="mt-4"),

            html.Div(id=ids.CLASSIFICATION_ROW),
            html.Div(id=ids.SCENARIO_BANNER),
            html.Div(id=ids.AI_NARRATIVE),

            # Collapsible: Performance Modeles
            html.Div([
                build_collapsible_button("Performance Modeles", ids.BTN_PERF),
                dbc.Collapse(
                    dcc.Loading(html.Div(id=ids.PERF_CONTENT), type="dot"),
                    id=ids.COLLAPSE_PERF, is_open=False,
                ),
            ], className="mt-3"),

            # Collapsible: Explainabilite SHAP
            html.Div([
                build_collapsible_button("Explainabilite SHAP", ids.BTN_SHAP),
                dbc.Collapse(
                    dcc.Loading(html.Div(id=ids.SHAP_CONTENT), type="dot"),
                    id=ids.COLLAPSE_SHAP, is_open=False,
                ),
            ], className="mt-3"),

            # Collapsible: Donnees & Export
            html.Div([
                build_collapsible_button("Donnees & Export", ids.BTN_EXPORT),
                dbc.Collapse(
                    html.Div([
                        # Apercu portefeuille (rempli par callback)
                        html.Div(id=ids.EXPORT_CONTENT),
                        # Boutons d'export (statiques — requis par Dash pour lier les callbacks)
                        dbc.Row([
                            dbc.Col(
                                dbc.Button(
                                    "Export Excel (8 feuilles)",
                                    id=ids.BTN_DL_EXCEL,
                                    n_clicks=0,
                                    outline=True, color="light", size="sm",
                                    className="w-100",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Rapport CRO (TXT)",
                                    id=ids.BTN_DL_CRO,
                                    n_clicks=0,
                                    outline=True, color="light", size="sm",
                                    className="w-100",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Synthese AI Analyst (TXT)",
                                    id=ids.BTN_DL_AI,
                                    n_clicks=0,
                                    outline=True, color="light", size="sm",
                                    className="w-100",
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Journal d'Audit (LaTeX)",
                                    id=ids.BTN_DL_LATEX,
                                    n_clicks=0,
                                    outline=True, color="light", size="sm",
                                    className="w-100",
                                ),
                                md=3,
                            ),
                        ], className="g-2 mt-3"),
                    ]),
                    id=ids.COLLAPSE_EXPORT, is_open=False,
                ),
            ], className="mt-3"),

            # Hidden download components
            dcc.Download(id=ids.DL_EXCEL),
            dcc.Download(id=ids.DL_CRO_TXT),
            dcc.Download(id=ids.DL_AI_TXT),
            dcc.Download(id=ids.DL_LATEX),

            # Toast container for notifications
            html.Div(id="toast-container"),

            # Footer
            html.Div(
                html.P(
                    "IFRS 9 Risk Cockpit v3.0 \u2014 M1 Finance Paris-Saclay",
                    style={
                        "textAlign": "center",
                        "color": "#64748B",
                        "fontSize": "0.75rem",
                        "padding": "2rem 0 1rem",
                    },
                ),
            ),

        ], className="main-content"),

        # ── Modals (at root level, outside main-content for proper centering) ──
        dbc.Modal([
            dbc.ModalHeader(
                dbc.ModalTitle("Private Equity \u2014 Detail"),
                close_button=True,
            ),
            dbc.ModalBody(
                html.Div(id=ids.MODAL_PE_BODY),
                style={"maxHeight": "85vh", "overflowY": "auto"},
            ),
        ], id=ids.MODAL_PE, size="xl", is_open=False, centered=True,
           className="modal-blur", backdrop=True, scrollable=True),

        dbc.Modal([
            dbc.ModalHeader(
                dbc.ModalTitle("Risque Credit \u2014 Detail"),
                close_button=True,
            ),
            dbc.ModalBody(
                html.Div(id=ids.MODAL_CREDIT_BODY),
                style={"maxHeight": "85vh", "overflowY": "auto"},
            ),
        ], id=ids.MODAL_CREDIT, size="xl", is_open=False, centered=True,
           className="modal-blur", backdrop=True, scrollable=True),
    ])
