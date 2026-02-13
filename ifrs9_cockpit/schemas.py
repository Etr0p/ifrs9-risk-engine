"""Schemas de validation Pandera pour les contrats de donnees IFRS 9.

Remplace les validations manuelles (_validate_data_contract) par des
schemas declaratifs auto-documentes. Chaque schema definit les types,
les bornes, et les contraintes statistiques des DataFrames d'entree
et de sortie du pipeline.

Usage :
    from ifrs9_cockpit.schemas import CreditInputSchema

    CreditInputSchema.validate(df_credit)  # raises SchemaError si invalide
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, Check

from ifrs9_cockpit.config import (
    ALLOWED_SECTORS,
    ALLOWED_LOAN_TYPES,
    CLIPPING_BOUNDS,
)


# ──────────────────────────────────────────────
# SCHEMA ENTREE CREDIT (CSV utilisateur ou DGP)
# ──────────────────────────────────────────────

CreditInputSchema = pa.DataFrameSchema(
    columns={
        "enterprise_id": Column(nullable=False),
        "sector": Column(
            str,
            Check.isin(sorted(ALLOWED_SECTORS)),
            nullable=False,
            description="Secteur entreprise (5 valeurs autorisees)",
        ),
        "loan_type": Column(
            str,
            Check.isin(sorted(ALLOWED_LOAN_TYPES)),
            nullable=False,
            description="Type de pret (Revolving ou Term)",
        ),
        "revenue": Column(
            float,
            Check.greater_than(0),
            nullable=False,
            description="Chiffre d'affaires en EUR",
        ),
        "ebitda": Column(
            float,
            nullable=False,
            description="EBITDA en EUR",
        ),
        "debt_ratio": Column(
            float,
            Check.in_range(
                CLIPPING_BOUNDS["debt_ratio"][0],
                CLIPPING_BOUNDS["debt_ratio"][1],
                include_min=True,
                include_max=True,
            ),
            nullable=False,
            coerce=True,
            description="Ratio d'endettement [0, 1.5]",
        ),
        "credit_score": Column(
            float,
            Check.in_range(
                CLIPPING_BOUNDS["credit_score"][0],
                CLIPPING_BOUNDS["credit_score"][1],
                include_min=True,
                include_max=True,
            ),
            nullable=False,
            coerce=True,
            description="Score de credit [300, 850]",
        ),
        "dpd": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            coerce=True,
            description="Days Past Due (jours de retard)",
        ),
        "collateral": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Valeur du collateral en EUR",
        ),
        "loan_amount": Column(
            float,
            Check.greater_than(0),
            nullable=False,
            description="Montant du pret en EUR",
        ),
        "utilization_rate": Column(
            float,
            Check.in_range(
                CLIPPING_BOUNDS["utilization_rate"][0],
                CLIPPING_BOUNDS["utilization_rate"][1],
                include_min=True,
                include_max=True,
            ),
            nullable=False,
            coerce=True,
            description="Taux d'utilisation [0, 1.2]",
        ),
        "default_flag": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Flag de defaut (0 = sain, 1 = defaut)",
        ),
        "pd_origination": Column(
            float,
            Check.in_range(0, 1, include_min=True, include_max=True),
            nullable=False,
            description="PD a l'origination [0, 1]",
        ),
    },
    coerce=True,
    strict=False,  # colonnes supplementaires autorisees
    name="CreditInputSchema",
    description="Schema d'entree du pipeline credit IFRS 9",
)


# ──────────────────────────────────────────────
# SCHEMA ENTREE PE
# ──────────────────────────────────────────────

PEInputSchema = pa.DataFrameSchema(
    columns={
        "enterprise_id": Column(nullable=False),
        "sector": Column(
            str,
            Check.isin(sorted(ALLOWED_SECTORS)),
            nullable=False,
        ),
        "revenue": Column(float, Check.greater_than(0), nullable=False),
        "ebitda": Column(float, nullable=False),
        "entry_multiple": Column(float, Check.greater_than(0), nullable=False),
        "leverage": Column(
            float,
            Check.in_range(0, 1, include_min=True, include_max=True),
            nullable=False,
        ),
        "vintage": Column(int, Check.greater_than_or_equal_to(2000), nullable=False, coerce=True),
        "holding_years": Column(float, Check.greater_than(0), nullable=False, coerce=True),
        "valuation_method": Column(str, nullable=False),
    },
    coerce=True,
    strict=False,
    name="PEInputSchema",
    description="Schema d'entree du pipeline PE IFRS 13",
)


# ──────────────────────────────────────────────
# SCHEMA SORTIE CREDIT (post-ECL)
# ──────────────────────────────────────────────

CreditResultSchema = pa.DataFrameSchema(
    columns={
        "enterprise_id": Column(nullable=False),
        "sector": Column(str, nullable=False),
        "pd_12m": Column(
            float,
            Check.in_range(0, 1, include_min=True, include_max=True),
            nullable=False,
            description="PD 12 mois [0, 1]",
        ),
        "pd_lifetime": Column(
            float,
            Check.in_range(0, 1, include_min=True, include_max=True),
            nullable=False,
            description="PD lifetime [0, 1]",
        ),
        "lgd": Column(
            float,
            Check.in_range(0, 1, include_min=True, include_max=True),
            nullable=False,
            description="LGD [0, 1]",
        ),
        "ead": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="EAD en EUR (>= 0)",
        ),
        "ecl_weighted": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="ECL ponderee multi-scenarios (>= 0)",
        ),
        "stage": Column(
            int,
            Check.isin([1, 2, 3]),
            nullable=False,
            description="Stage IFRS 9 (1, 2 ou 3)",
        ),
        "rwa_credit": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="RWA credit en EUR",
        ),
    },
    coerce=True,
    strict=False,
    name="CreditResultSchema",
    description="Schema de sortie du pipeline ECL credit",
)


# ──────────────────────────────────────────────
# SCHEMA SORTIE PE (post-calcul)
# ──────────────────────────────────────────────

PEResultSchema = pa.DataFrameSchema(
    columns={
        "enterprise_id": Column(nullable=False),
        "sector": Column(str, nullable=False),
        "nav": Column(float, nullable=False, description="Net Asset Value en EUR"),
        "delta_nav": Column(float, nullable=False, description="Variation NAV vs baseline"),
        "expected_loss_pe": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="EL PE en EUR (>= 0)",
        ),
        "risk_category": Column(
            str,
            Check.isin(["Performing", "Watchlist", "Distressed"]),
            nullable=False,
            description="Categorie de risque PE",
        ),
        "rwa_pe": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="RWA PE en EUR",
        ),
    },
    coerce=True,
    strict=False,
    name="PEResultSchema",
    description="Schema de sortie du pipeline PE IFRS 13",
)
