"""Package callbacks — decoupe modulaire du register().

Usage inchange :
    from ifrs9_cockpit.dashboard.callbacks import register
"""

from __future__ import annotations
from typing import Optional


def register(app, df_credit, df_pe, df_history, pd_suite, lgd_model, ead_model,
             *, df_balance_sheet=None, dataset_bundle=None):
    """Enregistre tous les callbacks sur l'app Dash.

    Args:
        app: Instance Dash.
        df_credit: DataFrame portefeuille credit.
        df_pe: DataFrame portefeuille PE.
        df_history: DataFrame historique macro.
        pd_suite: PDModelSuite pre-entrainee (3 familles).
        lgd_model: LGDModel calibre.
        ead_model: EADModel calibre.
        df_balance_sheet: DataFrame balance sheet 10 classes (optional).
        dataset_bundle: DatasetBundle with position DataFrames (optional).
    """
    from ifrs9_cockpit.dashboard.callbacks.scenario import register_scenario
    from ifrs9_cockpit.dashboard.callbacks.pipeline import register_pipeline
    from ifrs9_cockpit.dashboard.callbacks.ui_update import register_ui
    from ifrs9_cockpit.dashboard.callbacks.modals import register_modals
    from ifrs9_cockpit.dashboard.callbacks.collapsibles import register_collapsibles
    from ifrs9_cockpit.dashboard.callbacks.content_builders import register_content_builders
    from ifrs9_cockpit.dashboard.callbacks.exports import register_exports

    register_scenario(app)
    register_pipeline(
        app, df_credit, df_pe, df_history, pd_suite, lgd_model, ead_model,
        df_balance_sheet=df_balance_sheet, dataset_bundle=dataset_bundle,
    )
    register_ui(app)
    register_modals(app)
    register_collapsibles(app, pd_suite)
    register_content_builders(
        app, df_credit, df_pe, df_history, pd_suite,
        df_balance_sheet=df_balance_sheet,
    )
    register_exports(app, pd_suite)
