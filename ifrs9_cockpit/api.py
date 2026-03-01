"""API REST FastAPI — Interface decouplée pour le moteur IFRS 9.

Expose le moteur de calcul ECL/PE/CRO via des endpoints REST,
permettant de decoupler le frontend (Streamlit) du backend (engine).

Architecture cible :
    Client (Streamlit / React / Excel) <-> API (FastAPI) <-> Core Engine

Lancer avec :
    uvicorn ifrs9_cockpit.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints :
    GET  /health              — Healthcheck (liveness probe)
    GET  /models              — Info sur les modeles PD charges
    POST /compute             — Pipeline complet (ECL + PE + comparateur + CRO)
    POST /stress-test         — Stress test parametrique
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ifrs9_cockpit.utils.logging import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# PYDANTIC MODELS (Request / Response)
# ──────────────────────────────────────────────


class MacroParams(BaseModel):
    """Parametres macroeconomiques pour le stress test."""

    unemployment_rate: float = Field(7.5, ge=0, le=30, description="Taux de chomage (%)")
    gdp_growth: float = Field(1.2, ge=-10, le=10, description="Croissance PIB (%)")
    interest_rate: float = Field(3.5, ge=-2, le=15, description="Taux directeur BCE (%)")
    hpi_growth: float = Field(2.0, ge=-40, le=30, description="Variation prix immobiliers (%)")
    inflation_rate: float = Field(2.5, ge=-5, le=15, description="Inflation IPC (%)")
    unemployment_crisis: bool = Field(False, description="Mode crise chomage (PE)")


class ComputeRequest(BaseModel):
    """Requete pour le pipeline complet."""

    macro_params: MacroParams = Field(default_factory=MacroParams)
    selected_model: str = Field("LR_WoE", description="Modele PD a utiliser")
    pe_allocation_pct: int = Field(20, ge=0, le=40, description="Allocation PE (%)")
    rw_pe_selected: int = Field(250, description="Risk Weight PE CRR3")
    rst_target_ecl: Optional[float] = Field(None, description="Cible ECL RST (EUR)")


class KPISummary(BaseModel):
    """KPI de haut niveau du portefeuille."""

    ecl_total: float = Field(description="ECL total credit (EUR)")
    ecl_delta_pct: float = Field(description="Variation ECL vs base (%)")
    nav_total: float = Field(description="NAV PE total (EUR)")
    nav_drawdown: float = Field(description="Drawdown NAV PE (%)")
    raroc_credit: float = Field(description="RAROC credit (%)")
    risk_appetite_signal: str = Field(description="Signal risk appetite (vert/ambre/rouge)")
    n_clients: int = Field(description="Nombre de clients")
    stage_distribution: Dict[str, int] = Field(description="Distribution des stages")


class ComputeResponse(BaseModel):
    """Reponse du pipeline complet."""

    kpi: KPISummary
    computation_time_ms: float = Field(description="Temps de calcul (ms)")
    model_used: str
    macro_params: MacroParams


class ModelInfo(BaseModel):
    """Information sur un modele PD."""

    name: str
    auc: float
    gini: float
    brier: float
    n_features: int


class ModelsResponse(BaseModel):
    """Reponse avec les informations des modeles."""

    models: List[ModelInfo]
    pretrained: bool
    source: str


class HealthResponse(BaseModel):
    """Reponse du healthcheck."""

    status: str = "ok"
    version: str = "3.0"
    models_loaded: bool = False


# ──────────────────────────────────────────────
# APPLICATION FASTAPI
# ──────────────────────────────────────────────

app = FastAPI(
    title="IFRS 9 Risk Cockpit API",
    description="API REST pour le moteur de calcul ECL/PE/CRO IFRS 9",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Etat global (charge une seule fois)
_state: Dict = {}


def _get_state() -> Dict:
    """Charge les donnees et modeles au premier appel (lazy init)."""
    if "loaded" not in _state:
        logger.info("loading_state", status="starting")
        t0 = time.perf_counter()

        from ifrs9_cockpit.data.generator import generate_dataset
        from ifrs9_cockpit.models.pd_model import PDModelSuite
        from ifrs9_cockpit.models.lgd_model import LGDModel
        from ifrs9_cockpit.models.ead_model import EADModel

        # Charger donnees
        df_credit, df_pe, df_history, _ = generate_dataset()
        _state["df_credit"] = df_credit
        _state["df_pe"] = df_pe
        _state["df_history"] = df_history

        # Charger ou entrainer modeles PD
        pretrained_path = Path("ifrs9_cockpit/training/models/pd_suite.joblib")
        if pretrained_path.exists():
            pd_suite = PDModelSuite.load(str(pretrained_path))
            _state["pretrained"] = True
        else:
            pd_suite = PDModelSuite()
            pd_suite.fit(df_credit)
            _state["pretrained"] = False
        _state["pd_suite"] = pd_suite

        # Calibrer LGD & EAD
        lgd_model = LGDModel()
        lgd_model.fit(df_credit)
        ead_model = EADModel()
        ead_model.fit(df_credit)
        _state["lgd_model"] = lgd_model
        _state["ead_model"] = ead_model

        _state["loaded"] = True
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("state_loaded", elapsed_ms=f"{elapsed_ms:.0f}")

    return _state


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck (liveness probe pour Docker/K8s)."""
    return HealthResponse(
        status="ok",
        version="3.0",
        models_loaded="loaded" in _state,
    )


@app.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    """Retourne les informations sur les modeles PD charges."""
    state = _get_state()
    pd_suite = state["pd_suite"]

    models = []
    for name, result in pd_suite.results.items():
        metrics = result.metrics_test
        models.append(ModelInfo(
            name=name,
            auc=metrics.get("auc", 0.0),
            gini=metrics.get("gini", 0.0),
            brier=metrics.get("brier", 0.0),
            n_features=len(result.feature_importance) if result.feature_importance else 0,
        ))

    return ModelsResponse(
        models=models,
        pretrained=state.get("pretrained", False),
        source="joblib" if state.get("pretrained") else "synthetic_30k",
    )


@app.post("/compute", response_model=ComputeResponse)
def compute(request: ComputeRequest) -> ComputeResponse:
    """Execute le pipeline complet ECL + PE + comparateur + CRO."""
    t0 = time.perf_counter()
    state = _get_state()

    pd_suite = state["pd_suite"]
    lgd_model = state["lgd_model"]
    ead_model = state["ead_model"]
    df_credit = state["df_credit"]
    df_pe = state["df_pe"]

    # Verifier le modele demande
    if request.selected_model not in pd_suite.results:
        raise HTTPException(
            status_code=400,
            detail=f"Modele '{request.selected_model}' non disponible. "
                   f"Modeles disponibles : {list(pd_suite.results.keys())}",
        )

    macro = request.macro_params

    # Import des moteurs
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.engine.comparator import PortfolioComparator
    from ifrs9_cockpit.ai_analyst import CROAnalyst

    logger.info(
        "compute_started",
        model=request.selected_model,
        unemployment=macro.unemployment_rate,
        gdp=macro.gdp_growth,
    )

    # 1. ECL Credit
    pd_predictions = pd_suite.predict(df_credit)
    pd_current = pd_predictions[request.selected_model]
    pd_origination = df_credit["pd_origination"].to_numpy()

    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    result_base = ecl_calc.calculate(df_credit, pd_current, pd_origination)
    ecl_base_total = result_base["ecl_weighted"].sum()

    result_stressed = ecl_calc.calculate(
        df_credit, pd_current, pd_origination,
        unemployment_override=macro.unemployment_rate,
        gdp_override=macro.gdp_growth,
        interest_rate_override=macro.interest_rate,
        hpi_override=macro.hpi_growth,
        inflation_override=macro.inflation_rate,
    )

    # 2. PE IFRS 13
    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(
        df_pe,
        unemployment_override=macro.unemployment_rate,
        gdp_override=macro.gdp_growth,
        interest_rate_override=macro.interest_rate,
        hpi_override=macro.hpi_growth,
        inflation_override=macro.inflation_rate,
        unemployment_crisis=macro.unemployment_crisis,
    )

    # 3. Comparaison & RAROC
    comparator = PortfolioComparator(result_stressed, result_pe)
    raroc_eva = comparator.compute_raroc_eva()

    # 4. AI Analyst
    macro_dict = {
        "unemployment_rate": macro.unemployment_rate,
        "gdp_growth": macro.gdp_growth,
        "interest_rate": macro.interest_rate,
        "hpi_growth": macro.hpi_growth,
        "inflation_rate": macro.inflation_rate,
    }
    cro_analyst = CROAnalyst(
        result_stressed, result_pe, macro_dict,
        target_ecl=request.rst_target_ecl,
    )
    analytics_state = cro_analyst.analyze()

    # KPI
    ecl_total = result_stressed["ecl_weighted"].sum()
    ecl_delta = (ecl_total - ecl_base_total) / max(ecl_base_total, 1)

    nav_total = result_pe["nav"].sum()
    delta_nav = result_pe["delta_nav"].sum()
    nav_ref = nav_total - delta_nav
    drawdown = max(0, -delta_nav) / max(nav_ref, 1)

    raroc_row = raroc_eva.filter(
        (pl.col("sector") == "Total") & (pl.col("canal") == "Credit")
    )
    raroc_val = float(raroc_row["raroc"][0]) if len(raroc_row) > 0 else 0.0

    ra = analytics_state.risk_appetite_matrix
    ra_rouge = int((ra["signal"] == "rouge").sum()) if ra is not None and len(ra) > 0 else 0
    ra_signal = "rouge" if ra_rouge > 3 else "ambre" if ra_rouge > 1 else "vert"

    _vc = result_stressed["stage"].value_counts()
    stage_counts = dict(zip(_vc["stage"].to_list(), _vc["count"].to_list()))
    stage_dist = {f"Stage {k}": int(v) for k, v in sorted(stage_counts.items())}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("compute_completed", elapsed_ms=f"{elapsed_ms:.0f}", ecl_total=f"{ecl_total:.0f}")

    return ComputeResponse(
        kpi=KPISummary(
            ecl_total=ecl_total,
            ecl_delta_pct=ecl_delta,
            nav_total=nav_total,
            nav_drawdown=drawdown,
            raroc_credit=raroc_val,
            risk_appetite_signal=ra_signal,
            n_clients=len(df_credit),
            stage_distribution=stage_dist,
        ),
        computation_time_ms=elapsed_ms,
        model_used=request.selected_model,
        macro_params=macro,
    )


@app.post("/stress-test", response_model=ComputeResponse)
def stress_test(request: ComputeRequest) -> ComputeResponse:
    """Alias pour /compute — clarte semantique pour les clients."""
    return compute(request)
