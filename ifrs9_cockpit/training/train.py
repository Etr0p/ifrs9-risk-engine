"""Script d'entrainement offline des modeles PD.

Entraine les 3 modeles (LR_WoE, TabNet, XGBoost) sur un dataset utilisateur
ou synthetique, puis serialise la suite pour chargement rapide par le dashboard.

Usage :
    # Avec donnees synthetiques (1.5M par defaut — Gold Standard)
    python -m ifrs9_cockpit.training.train

    # Avec CSV utilisateur
    python -m ifrs9_cockpit.training.train --data ifrs9_cockpit/training/data/portefeuille.csv

    # Avec taille custom
    python -m ifrs9_cockpit.training.train --n-clients 500000

    # Chemin de sortie custom
    python -m ifrs9_cockpit.training.train --output mon_modele.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le repertoire racine au path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ifrs9_cockpit.config import (
    ALLOWED_LOAN_TYPES,
    ALLOWED_SECTORS,
    CATEGORICAL_FEATURES,
    CLIPPING_BOUNDS,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
)
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.schemas import CreditInputSchema
from ifrs9_cockpit.utils.logging import get_logger

logger = get_logger(__name__)


# Colonnes minimales requises dans un CSV utilisateur
# Les features engineered sont calculees automatiquement par le pipeline
REQUIRED_COLUMNS = frozenset(
    [TARGET]
    + [f for f in NUMERICAL_FEATURES if f not in ENGINEERED_FEATURES]
    + CATEGORICAL_FEATURES
)

DEFAULT_OUTPUT = Path("ifrs9_cockpit/training/models/pd_suite.joblib")
DEFAULT_N_CLIENTS = 1_500_000


def _validate_data_contract(df: pd.DataFrame) -> None:
    """Valide le contrat de donnees sur un DataFrame CSV utilisateur.

    Utilise le schema Pandera CreditInputSchema pour la validation declarative,
    complete par des warnings sur les bornes et les NaN.

    Args:
        df: DataFrame a valider.

    Raises:
        ValueError: Si des colonnes requises manquent ou si des enums sont invalides.
        pandera.errors.SchemaError: Si le schema Pandera est viole.
    """
    # 1. Colonnes requises (verif rapide avant Pandera)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {sorted(missing)}\n"
            f"Colonnes trouvees : {sorted(df.columns)}\n"
            f"Colonnes requises : {sorted(REQUIRED_COLUMNS)}"
        )

    # 2. Validation Pandera (types, enums, bornes — declaratif)
    try:
        CreditInputSchema.validate(df, lazy=True)
        logger.info("pandera_validation_passed", n_rows=len(df))
    except Exception as e:
        logger.error("pandera_validation_failed", error=str(e))
        raise

    # 3. Bornes numeriques (warning — le clipping corrige automatiquement)
    for col, (lo, hi) in CLIPPING_BOUNDS.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if lo is not None and col_min < lo:
            logger.warning("bounds_violation", column=col, min=f"{col_min:.4f}", bound=lo, action="will_be_clipped")
        if hi is not None and col_max > hi:
            logger.warning("bounds_violation", column=col, max=f"{col_max:.4f}", bound=hi, action="will_be_clipped")

    # 4. NaN > 5% (warning)
    for col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.05:
            logger.warning("high_nan_rate", column=col, nan_pct=f"{nan_pct:.1%}")


def _load_csv(path: str) -> pd.DataFrame:
    """Charge et valide un CSV utilisateur.

    Args:
        path: Chemin du fichier CSV.

    Returns:
        DataFrame valide.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si des colonnes requises sont manquantes ou enums invalides.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(p)
    _validate_data_contract(df)

    logger.info("csv_loaded", n_rows=len(df), n_cols=len(df.columns), default_rate=f"{df[TARGET].mean():.2%}")
    return df


def _generate_synthetic(n_clients: int) -> pd.DataFrame:
    """Genere un dataset synthetique via le DGP v4.5.

    Utilise seed=42 pour l'entrainement (different du portfolio seed=123).

    Args:
        n_clients: Nombre de clients a generer.

    Returns:
        DataFrame credit synthetique.
    """
    from ifrs9_cockpit.synthetic_generator_v4 import (
        generate_dataset as _v4_generate,
    )

    logger.info("generating_synthetic", n_clients=n_clients, dgp="v4.5", seed=42)
    df_credit, _, _ = _v4_generate(n_clients=n_clients, seed=42)
    logger.info("synthetic_generated", default_rate=f"{df_credit[TARGET].mean():.2%}")
    return df_credit


def main() -> None:
    """Point d'entree principal du script d'entrainement."""
    parser = argparse.ArgumentParser(
        description="Entrainement offline des modeles PD (LR_WoE / TabNet / XGBoost)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Chemin du CSV utilisateur. Si absent, genere des donnees synthetiques.",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=DEFAULT_N_CLIENTS,
        help=f"Nombre de clients synthetiques (defaut: {DEFAULT_N_CLIENTS:,}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Chemin de sortie du modele (defaut: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Modeles a entrainer (comma-separated). Ex: LR_WoE,XGBoost. Defaut: tous.",
    )
    args = parser.parse_args()

    logger.info("training_started", pipeline="IFRS 9 COCKPIT — Entrainement Offline PD")

    # 1. Chargement des donnees
    logger.info("stage_started", stage="1/4", action="loading_data")
    t0 = time.perf_counter()
    if args.data:
        df = _load_csv(args.data)
    else:
        df = _generate_synthetic(args.n_clients)
    t_load = time.perf_counter() - t0
    logger.info("stage_completed", stage="1/4", elapsed_s=f"{t_load:.1f}")

    # 2. Entrainement
    models_to_train = tuple(args.models.split(",")) if args.models else None
    model_names = ", ".join(models_to_train) if models_to_train else "LR_WoE, TabNet, XGBoost"
    logger.info("stage_started", stage="2/4", action="training", models=model_names)
    t1 = time.perf_counter()
    suite = PDModelSuite()
    suite.fit(df, models=models_to_train)
    t_train = time.perf_counter() - t1
    logger.info("stage_completed", stage="2/4", elapsed_s=f"{t_train:.1f}")

    # 3. Benchmark
    logger.info("stage_started", stage="3/4", action="benchmark")
    comparison = suite.get_comparison_table()
    print(comparison.to_string(index=False))  # Benchmark table to stdout

    # Feature importance TabNet
    tabnet_result = suite.results.get("TabNet")
    if tabnet_result and tabnet_result.feature_importance:
        sorted_fi = sorted(
            tabnet_result.feature_importance.items(),
            key=lambda x: -x[1],
        )
        for feat, imp in sorted_fi[:10]:
            logger.debug("tabnet_feature_importance", feature=feat, importance=f"{imp:.4f}")

    # 4. Sauvegarde
    logger.info("stage_started", stage="4/4", action="saving")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suite.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("model_saved", path=str(output_path), size_mb=f"{size_mb:.1f}")

    # Resume
    total_time = time.perf_counter() - t0
    logger.info(
        "training_completed",
        n_rows=len(df),
        total_time_s=f"{total_time:.1f}",
        output=str(output_path),
    )


if __name__ == "__main__":
    main()
