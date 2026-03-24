"""Script d'entrainement offline — PD + LGD/EAD + Gouvernance + GFlowNet.

Entraine les 3 modeles PD (LR_WoE, TabNet, XGBoost), calibre LGD/EAD,
pre-calcule les artefacts de gouvernance (Conformal, Sobol, RMT, Signatures,
TDA, HMM, GFlowNet), puis serialise le tout pour chargement rapide.

Usage :
    # Complet (1.5M synthetiques)
    python -m ifrs9_cockpit.training.train

    # Taille custom
    python -m ifrs9_cockpit.training.train --n-clients 2000

    # CSV utilisateur
    python -m ifrs9_cockpit.training.train --data portefeuille.csv

    # Chemin de sortie custom
    python -m ifrs9_cockpit.training.train --output mon_modele.joblib

    # Sans gouvernance (PD seulement)
    python -m ifrs9_cockpit.training.train --skip-governance
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

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
    CONSUMER_NUMERICAL_FEATURES,
    CONSUMER_CATEGORICAL_FEATURES,
    CONSUMER_ENGINEERED_FEATURES,
    MORTGAGE_NUMERICAL_FEATURES,
    MORTGAGE_CATEGORICAL_FEATURES,
    MORTGAGE_ENGINEERED_FEATURES,
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
DEFAULT_CONSUMER_OUTPUT = Path("ifrs9_cockpit/training/models/consumer_pd_suite.joblib")
DEFAULT_MORTGAGE_OUTPUT = Path("ifrs9_cockpit/training/models/mortgage_pd_suite.joblib")
DEFAULT_GOVERNANCE_OUTPUT = Path("ifrs9_cockpit/training/models/governance_suite.joblib")
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
    from ifrs9_cockpit.data.generator import generate_dataset as _generate

    logger.info("generating_synthetic", n_clients=n_clients, dgp="v4.5", seed=42)
    df_credit, _, _, _ = _generate(n_clients=n_clients, seed=42)
    logger.info("synthetic_generated", default_rate=f"{df_credit[TARGET].mean():.2%}")
    return df_credit


def _load_consumer_data() -> Optional[pd.DataFrame]:
    """Charge les donnees consumer Lending Club pour entrainement.

    Returns:
        DataFrame consumer ou None si le parquet est absent.
    """
    try:
        from ifrs9_cockpit.synthetic_generator.consumer_positions import load_consumer_data
        df = load_consumer_data(split="train")
        logger.info("consumer_data_loaded", n_rows=len(df), split="train")
        return df
    except FileNotFoundError:
        logger.warning("consumer_data_absent", msg="consumer_credit.parquet not found, skipping consumer suite")
        return None
    except Exception as e:
        logger.warning("consumer_data_error", error=str(e))
        return None


def _generate_mortgage_training_data() -> pd.DataFrame:
    """Genere 50k positions hypothecaires pour entrainement PD.

    Returns:
        DataFrame mortgage avec ~15 features + default_flag.
    """
    from ifrs9_cockpit.synthetic_generator.mortgage_positions import generate_mortgage_training_data
    df = generate_mortgage_training_data(n_positions=50_000, seed=42)
    logger.info("mortgage_data_generated", n_rows=len(df), default_rate=f"{df[TARGET].mean():.2%}")
    return df


def _train_consumer_suite(df: pd.DataFrame, models_to_train: Optional[tuple] = None) -> PDModelSuite:
    """Entraine une suite PD consumer sur les donnees Lending Club.

    Args:
        df: DataFrame consumer.
        models_to_train: Tuple de modeles (defaut: LR_WoE, TabNet, XGBoost).

    Returns:
        PDModelSuite entrainee.
    """
    models = models_to_train or ("LR_WoE", "TabNet", "XGBoost")
    suite = PDModelSuite(
        seed=42,
        numerical_features=CONSUMER_NUMERICAL_FEATURES + CONSUMER_ENGINEERED_FEATURES,
        categorical_features=CONSUMER_CATEGORICAL_FEATURES,
        available_models=models,
        clipping_bounds={
            "credit_score": (300.0, 850.0),
            "dti": (0.0, 100.0),
            "utilization_rate": (0.0, 1.5),
        },
        tabnet_variant="light",
    )
    suite.fit(df, models=models)
    return suite


def _train_mortgage_suite(df: pd.DataFrame, models_to_train: Optional[tuple] = None) -> PDModelSuite:
    """Entraine une suite PD mortgage sur les positions synthetiques.

    Args:
        df: DataFrame mortgage.
        models_to_train: Tuple de modeles (defaut: LR_WoE, TabNet, XGBoost).

    Returns:
        PDModelSuite entrainee.
    """
    models = models_to_train or ("LR_WoE", "TabNet", "XGBoost")
    suite = PDModelSuite(
        seed=42,
        numerical_features=MORTGAGE_NUMERICAL_FEATURES + MORTGAGE_ENGINEERED_FEATURES,
        categorical_features=MORTGAGE_CATEGORICAL_FEATURES,
        available_models=models,
        clipping_bounds={
            "ltv": (0.0, 1.5),
            "dti": (0.0, 1.0),
        },
        tabnet_variant="light",
    )
    suite.fit(df, models=models)
    return suite


def _train_governance(df: pd.DataFrame, suite: PDModelSuite) -> dict:
    """Pre-calcule tous les artefacts de gouvernance.

    Args:
        df: DataFrame credit (pour calibration gouvernance).
        suite: PDModelSuite entrainee (pour PD predictions).

    Returns:
        Dict d'artefacts pret a serialiser.
    """
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.signatures import compute_macro_signatures
    from ifrs9_cockpit.engine.tda import compute_macro_fragility
    from ifrs9_cockpit.engine.rmt import denoise_covariance
    from ifrs9_cockpit.engine.hmm_regime import GaussianHMM
    from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis
    from ifrs9_cockpit.config import (
        MACRO_COVARIANCE,
        MACRO_HISTORY_BASELINE,
        MACRO_VARIABLES_ORDER,
    )

    artifacts = {}

    # ── Stage 3: LGD + EAD calibration ──
    logger.info("stage_started", stage="3/6", action="lgd_ead_calibration")
    t3 = time.perf_counter()

    lgd_model = LGDModel()
    lgd_model.fit(df)
    ead_model = EADModel()
    ead_model.fit(df)
    artifacts["lgd_model"] = lgd_model
    artifacts["ead_model"] = ead_model

    logger.info("stage_completed", stage="3/6", elapsed_s=f"{time.perf_counter() - t3:.1f}")

    # ── Stage 4: Governance engines ──
    logger.info("stage_started", stage="4/6", action="governance_engines")
    t4 = time.perf_counter()

    # 4a. Path Signatures (ordre 3 offline pour richesse, ordre 2 pour display)
    try:
        sig_result_3 = compute_macro_signatures(MACRO_HISTORY_BASELINE, order=3)
        sig_result_2 = compute_macro_signatures(MACRO_HISTORY_BASELINE, order=2)
        artifacts["signatures"] = sig_result_2      # display (heatmap lisible)
        artifacts["signatures_order3"] = sig_result_3  # stockage riche
        logger.info("governance_sub", engine="signatures",
                     n_features_display=sig_result_2.n_features,
                     n_features_full=sig_result_3.n_features)
    except Exception as e:
        logger.warning("governance_skipped", engine="signatures", error=str(e))

    # 4b. TDA Fragility
    try:
        tda_result = compute_macro_fragility(MACRO_HISTORY_BASELINE, window=24)
        artifacts["tda"] = tda_result
        logger.info("governance_sub", engine="tda", fragility=f"{tda_result.fragility_index:.3f}")
    except Exception as e:
        logger.warning("governance_skipped", engine="tda", error=str(e))

    # 4c. RMT Denoising
    try:
        cov_matrix = np.array(MACRO_COVARIANCE)
        rmt_result = denoise_covariance(cov_matrix, n_observations=60)
        artifacts["rmt"] = rmt_result
        logger.info("governance_sub", engine="rmt", n_signal=rmt_result.n_signal, n_noise=rmt_result.n_noise)
    except Exception as e:
        logger.warning("governance_skipped", engine="rmt", error=str(e))

    # 4d. HMM Regime (fit on baseline history, serialize full object)
    try:
        obs = np.column_stack([
            MACRO_HISTORY_BASELINE[v]
            for v in MACRO_VARIABLES_ORDER
        ])  # (60, 5)
        hmm = GaussianHMM(n_regimes=3, seed=42)
        hmm.fit(obs)
        artifacts["hmm"] = hmm
        logger.info("governance_sub", engine="hmm", fitted=True)
    except Exception as e:
        logger.warning("governance_skipped", engine="hmm", error=str(e))

    # PD predictions (shared by sobol)
    try:
        pd_pred = suite.predict_active(df)
    except Exception:
        # Fallback: use pd_origination column or synthetic PDs
        pd_pred = df["pd_origination"].values if "pd_origination" in df.columns else np.random.default_rng(42).uniform(0.001, 0.15, len(df))
    y_true = df[TARGET].values

    # 4e. Sobol (N=512 offline — better quality than inline N=256)
    try:
        import polars as pl
        ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
        # Subsample for speed (Sobol calls ecl_fn N*(D+2) times)
        _MAX_SOBOL = 5000
        df_sobol_pd = df.head(_MAX_SOBOL) if len(df) > _MAX_SOBOL else df
        df_sobol = pl.from_pandas(df_sobol_pd) if isinstance(df_sobol_pd, pd.DataFrame) else df_sobol_pd
        pd_current = pd_pred[:_MAX_SOBOL] if len(pd_pred) > _MAX_SOBOL else pd_pred
        pd_origination = pd_current * 0.8

        def ecl_fn(params):
            try:
                r = ecl_calc.calculate(
                    df_sobol, pd_current, pd_origination,
                    unemployment_override=params.get("unemployment_rate"),
                    gdp_override=params.get("gdp_growth"),
                    interest_rate_override=params.get("interest_rate"),
                    hpi_override=params.get("hpi_growth"),
                    inflation_override=params.get("inflation_rate"),
                )
                return float(r["ecl_weighted"].sum())
            except Exception:
                return 0.0

        sobol_result = sobol_analysis(ecl_fn, n_samples=512, seed=42)
        artifacts["sobol"] = sobol_result
        logger.info("governance_sub", engine="sobol", n_samples=512)
    except Exception as e:
        logger.warning("governance_skipped", engine="sobol", error=str(e))

    logger.info("stage_completed", stage="4/6", elapsed_s=f"{time.perf_counter() - t4:.1f}")

    # ── Stage 5: GFlowNet pre-training ──
    logger.info("stage_started", stage="5/6", action="gflownet_pretraining")
    t5 = time.perf_counter()

    try:
        from ifrs9_cockpit.engine.gflownet import DualGFlowNet

        ecl_total = 1e9  # Default ECL estimate for surrogate

        def ecl_surrogate(params):
            u = params.get("unemployment_rate", 7.5)
            g = params.get("gdp_growth", 1.2)
            r = params.get("interest_rate", 3.5)
            h = params.get("hpi_growth", 2.0)
            return ecl_total * (1 + 0.12 * (u - 7.5) - 0.06 * (g - 1.2) + 0.04 * (r - 3.5) - 0.03 * (h - 2.0))

        dual = DualGFlowNet(seed=42)
        gfn_result = dual.run(
            regime="contraction",
            ecl_surrogate=ecl_surrogate,
            breach_threshold=ecl_total * 1.5,
            n_samples=100,
            n_train_steps=200,
        )
        # Serialize the full DualGFlowNet object (no _extract_weights API)
        artifacts["gflownet"] = dual
        logger.info("governance_sub", engine="gflownet", n_samples=100, n_steps=200)
    except Exception as e:
        logger.warning("governance_skipped", engine="gflownet", error=str(e))

    logger.info("stage_completed", stage="5/6", elapsed_s=f"{time.perf_counter() - t5:.1f}")

    return artifacts


def main() -> None:
    """Point d'entree principal du script d'entrainement."""
    parser = argparse.ArgumentParser(
        description="Entrainement offline — PD + LGD/EAD + Gouvernance + GFlowNet",
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
        help=f"Chemin de sortie du modele PD (defaut: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--output-governance",
        type=str,
        default=str(DEFAULT_GOVERNANCE_OUTPUT),
        help=f"Chemin de sortie gouvernance (defaut: {DEFAULT_GOVERNANCE_OUTPUT}).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Modeles a entrainer (comma-separated). Ex: LR_WoE,XGBoost. Defaut: tous.",
    )
    parser.add_argument(
        "--skip-governance",
        action="store_true",
        help="Sauter les stages gouvernance (PD seulement).",
    )
    parser.add_argument(
        "--skip-consumer",
        action="store_true",
        help="Sauter l'entrainement consumer PD suite.",
    )
    parser.add_argument(
        "--skip-mortgage",
        action="store_true",
        help="Sauter l'entrainement mortgage PD suite.",
    )
    args = parser.parse_args()

    n_stages = "4" if args.skip_governance else "6"
    logger.info("training_started", pipeline="IFRS 9 COCKPIT — Entrainement Offline")

    # 1. Chargement des donnees
    logger.info("stage_started", stage=f"1/{n_stages}", action="loading_data")
    t0 = time.perf_counter()
    if args.data:
        df = _load_csv(args.data)
    else:
        df = _generate_synthetic(args.n_clients)
    t_load = time.perf_counter() - t0
    logger.info("stage_completed", stage=f"1/{n_stages}", elapsed_s=f"{t_load:.1f}")

    # 2. Entrainement PD
    models_to_train = tuple(args.models.split(",")) if args.models else None
    model_names = ", ".join(models_to_train) if models_to_train else "LR_WoE, TabNet, XGBoost"
    logger.info("stage_started", stage=f"2/{n_stages}", action="training", models=model_names)
    t1 = time.perf_counter()
    suite = PDModelSuite()
    suite.fit(df, models=models_to_train)
    t_train = time.perf_counter() - t1
    logger.info("stage_completed", stage=f"2/{n_stages}", elapsed_s=f"{t_train:.1f}")

    # Benchmark
    comparison = suite.get_comparison_table()
    print(comparison.to_pandas().to_string(index=False))

    # Feature importance TabNet
    tabnet_result = suite.results.get("TabNet")
    if tabnet_result and tabnet_result.feature_importance:
        sorted_fi = sorted(
            tabnet_result.feature_importance.items(),
            key=lambda x: -x[1],
        )
        for feat, imp in sorted_fi[:10]:
            logger.debug("tabnet_feature_importance", feature=feat, importance=f"{imp:.4f}")

    # 2b. Consumer PD Suite (optional)
    consumer_suite = None
    if not args.skip_consumer:
        logger.info("stage_started", stage="2b", action="consumer_pd_training")
        t2b = time.perf_counter()
        df_consumer = _load_consumer_data()
        if df_consumer is not None:
            consumer_suite = _train_consumer_suite(df_consumer)
            t_consumer = time.perf_counter() - t2b
            logger.info("stage_completed", stage="2b", elapsed_s=f"{t_consumer:.1f}")
            print("\n--- Consumer PD Suite ---")
            print(consumer_suite.get_comparison_table().to_pandas().to_string(index=False))
        else:
            logger.info("stage_skipped", stage="2b", reason="no_parquet")

    # 2c. Mortgage PD Suite (optional)
    mortgage_suite = None
    if not args.skip_mortgage:
        logger.info("stage_started", stage="2c", action="mortgage_pd_training")
        t2c = time.perf_counter()
        df_mortgage = _generate_mortgage_training_data()
        mortgage_suite = _train_mortgage_suite(df_mortgage)
        t_mortgage = time.perf_counter() - t2c
        logger.info("stage_completed", stage="2c", elapsed_s=f"{t_mortgage:.1f}")
        print("\n--- Mortgage PD Suite ---")
        print(mortgage_suite.get_comparison_table().to_pandas().to_string(index=False))

    # 3-5. Governance (optional)
    governance_artifacts = None
    if not args.skip_governance:
        governance_artifacts = _train_governance(df, suite)

    # Final. Serialization
    logger.info("stage_started", stage="final", action="saving")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suite.save(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("model_saved", path=str(output_path), size_mb=f"{size_mb:.1f}", suite="corporate")

    if consumer_suite is not None:
        consumer_path = DEFAULT_CONSUMER_OUTPUT
        consumer_path.parent.mkdir(parents=True, exist_ok=True)
        consumer_suite.save(str(consumer_path))
        cs_mb = consumer_path.stat().st_size / (1024 * 1024)
        logger.info("model_saved", path=str(consumer_path), size_mb=f"{cs_mb:.1f}", suite="consumer")

    if mortgage_suite is not None:
        mortgage_path = DEFAULT_MORTGAGE_OUTPUT
        mortgage_path.parent.mkdir(parents=True, exist_ok=True)
        mortgage_suite.save(str(mortgage_path))
        ms_mb = mortgage_path.stat().st_size / (1024 * 1024)
        logger.info("model_saved", path=str(mortgage_path), size_mb=f"{ms_mb:.1f}", suite="mortgage")

    if governance_artifacts is not None:
        import joblib

        gov_path = Path(args.output_governance)
        gov_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(governance_artifacts, str(gov_path))
        gov_size_mb = gov_path.stat().st_size / (1024 * 1024)
        logger.info("governance_saved", path=str(gov_path), size_mb=f"{gov_size_mb:.1f}",
                     keys=sorted(governance_artifacts.keys()))

    # Summary
    print("\n" + "=" * 65)
    print("TRAINING SUMMARY")
    print("=" * 65)
    _suites = [("Corporate", suite, str(output_path))]
    if consumer_suite is not None:
        _suites.append(("Consumer", consumer_suite, str(DEFAULT_CONSUMER_OUTPUT)))
    if mortgage_suite is not None:
        _suites.append(("Mortgage", mortgage_suite, str(DEFAULT_MORTGAGE_OUTPUT)))
    for name, s, path in _suites:
        best_auc = max(r.metrics_test["auc"] for r in s.results.values())
        models_str = "/".join(s.results.keys())
        print(f"  {name:12s}  Models: {models_str:20s}  AUC(best): {best_auc:.4f}  File: {path}")
    print("=" * 65)

    total_time = time.perf_counter() - t0
    logger.info(
        "training_completed",
        n_rows=len(df),
        total_time_s=f"{total_time:.1f}",
        output=str(output_path),
    )


if __name__ == "__main__":
    main()
