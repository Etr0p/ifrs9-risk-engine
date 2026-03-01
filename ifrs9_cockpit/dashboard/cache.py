"""Cache serveur pour le dashboard Dash -- remplace st.cache_data / st.cache_resource.

Architecture de cache a 3 niveaux :
    1. load_data() / train_pd_models() / train_lgd_ead()
       -> functools.lru_cache(maxsize=1) : donnees et modeles charges une seule fois.
    2. pipeline results cache
       -> dict thread-safe avec cle = hash(macro_params + model + PE config).
       -> max 10 entries (LRU simple par eviction FIFO).
    3. SHAP values cache
       -> dict cle = model_name + n_samples, evite le recalcul couteux.

Thread-safety : le pipeline cache est protege par un threading.Lock car Dash
peut servir plusieurs requetes en parallele (multi-worker ou callbacks async).

Remplacement des caches Streamlit :
    - st.cache_data   -> functools.lru_cache (donnees immutables)
    - st.cache_resource -> functools.lru_cache (modeles, objets lourds)
    - st.session_state -> dcc.Store + pipeline_cache (cote serveur)
"""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl

from ifrs9_cockpit.utils.dataset_bundle import DatasetBundle


# ──────────────────────────────────────────────
# DISK CACHE (survit aux reloads Werkzeug)
# ──────────────────────────────────────────────
_CACHE_DIR = Path(tempfile.gettempdir()) / "ifrs9_cockpit_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_TTL = 3600  # 1h — invalider si le process a crashe depuis longtemps


def _disk_cache_get(key: str):
    """Charge un objet du cache disque (joblib compress). None si absent/expire."""
    path = _CACHE_DIR / f"{key}.z"
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > _CACHE_TTL:
            path.unlink(missing_ok=True)
            return None
        import joblib
        return joblib.load(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _disk_cache_set(key: str, obj):
    """Sauvegarde un objet sur disque (joblib compress lz4)."""
    path = _CACHE_DIR / f"{key}.z"
    try:
        import joblib
        joblib.dump(obj, path, compress=("lz4", 1))
    except Exception:
        try:
            import joblib
            joblib.dump(obj, path, compress=3)  # fallback zlib
        except Exception:
            pass


# ──────────────────────────────────────────────
# PIPELINE RESULTS CACHE (thread-safe)
# ──────────────────────────────────────────────
_pipeline_lock = threading.Lock()
_pipeline_cache: Dict[str, Dict[str, Any]] = {}


def pipeline_key(
    macro_params: dict,
    selected_model: str,
    pe_alloc: int,
    rw_pe: int,
    rst_ecl: float | None = None,
) -> str:
    """Hash deterministe des inputs pipeline pour cle de cache.

    Serialise tous les parametres en JSON trie puis retourne un hash MD5.
    Le MD5 n'est pas utilise pour la securite mais uniquement comme cle
    de cache compacte et deterministe.

    Args:
        macro_params: Dictionnaire des 5 variables macro (sliders).
        selected_model: Nom du modele PD actif (LR_WoE, TabNet, XGBoost).
        pe_alloc: Allocation PE en pourcentage (0-40).
        rw_pe: Risk Weight PE CRR3 (190, 250, 400).
        rst_ecl: Cible ECL reverse stress test (EUR) ou None.

    Returns:
        Hash MD5 hexadecimal (32 caracteres).
    """
    payload = json.dumps(
        {
            **macro_params,
            "model": selected_model,
            "pe": pe_alloc,
            "rw": rw_pe,
            "rst": rst_ecl,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def get_cached_pipeline(key: str) -> Optional[Dict[str, Any]]:
    """Recupere les resultats pipeline du cache serveur.

    Args:
        key: Cle MD5 retournee par pipeline_key().

    Returns:
        Dictionnaire des resultats pipeline, ou None si absent.
    """
    with _pipeline_lock:
        return _pipeline_cache.get(key)


def set_cached_pipeline(key: str, results: Dict[str, Any]) -> None:
    """Stocke les resultats pipeline dans le cache serveur.

    Politique d'eviction FIFO : si le cache depasse 10 entries,
    la plus ancienne est supprimee (dict Python 3.7+ preserve l'ordre).

    Args:
        key: Cle MD5 retournee par pipeline_key().
        results: Dictionnaire des resultats pipeline a stocker.
    """
    with _pipeline_lock:
        # Garder max 10 entries pour limiter la memoire
        if len(_pipeline_cache) > 10:
            oldest_key = next(iter(_pipeline_cache))
            del _pipeline_cache[oldest_key]
        _pipeline_cache[key] = results


# ──────────────────────────────────────────────
# DATA LOADING (lru_cache — charge une seule fois)
# ──────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def load_data() -> DatasetBundle:
    """Genere et cache le dataset (credit + PE + historique + balance sheet).

    Premier appel : genere puis sauvegarde sur disque.
    Reloads suivants : charge depuis le disque (~0.3s au lieu de ~1.1s).

    Returns:
        DatasetBundle with (df_credit, df_pe, df_history, df_balance_sheet)
        and 8 position DataFrames. Supports tuple unpacking for backward compat.
    """
    cached = _disk_cache_get("dataset")
    if cached is not None:
        print("  [cache] Dataset charge depuis le disque")
        return cached
    from ifrs9_cockpit.data.generator import generate_dataset
    result = generate_dataset()
    _disk_cache_set("dataset", result)
    return result


# ──────────────────────────────────────────────
# MODEL TRAINING / LOADING (lru_cache)
# ──────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def train_pd_models():
    """Charge ou entraine les modeles PD.

    Ordre de priorite :
        1. Modele pre-entraine (training/models/pd_suite.joblib)
        2. Cache disque (survit aux reloads)
        3. Fallback : entrainer sur donnees synthetiques

    Returns:
        PDModelSuite pre-entrainee ou fraichement entrainee.
    """
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    pretrained = Path("ifrs9_cockpit/training/models/pd_suite.joblib")
    if pretrained.exists():
        try:
            return PDModelSuite.load(str(pretrained))
        except Exception:
            pass
    cached = _disk_cache_get("pd_suite")
    if cached is not None:
        print("  [cache] Modeles PD charges depuis le disque")
        return cached
    df_credit, _, _, _ = load_data()
    suite = PDModelSuite()
    suite.fit(df_credit)
    _disk_cache_set("pd_suite", suite)
    return suite


@functools.lru_cache(maxsize=1)
def train_lgd_ead():
    """Calibre et cache les modeles LGD et EAD.

    Returns:
        Tuple (LGDModel, EADModel) calibres sur les donnees credit.
    """
    cached = _disk_cache_get("lgd_ead")
    if cached is not None:
        print("  [cache] LGD & EAD charges depuis le disque")
        return cached
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel
    df_credit, _, _, _ = load_data()
    lgd = LGDModel()
    lgd.fit(df_credit)
    ead = EADModel()
    ead.fit(df_credit)
    _disk_cache_set("lgd_ead", (lgd, ead))
    return lgd, ead


# ──────────────────────────────────────────────
# GOVERNANCE ARTIFACTS (lru_cache — charge une seule fois)
# ──────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def load_consumer_pd_suite():
    """Charge la suite PD consumer pre-entrainee.

    Returns:
        PDModelSuite consumer ou None si le fichier n'existe pas.
    """
    path = Path("ifrs9_cockpit/training/models/consumer_pd_suite.joblib")
    if path.exists():
        try:
            from ifrs9_cockpit.models.pd_model import PDModelSuite
            return PDModelSuite.load(str(path))
        except Exception:
            return None
    return None


@functools.lru_cache(maxsize=1)
def load_mortgage_pd_suite():
    """Charge la suite PD mortgage pre-entrainee.

    Returns:
        PDModelSuite mortgage ou None si le fichier n'existe pas.
    """
    path = Path("ifrs9_cockpit/training/models/mortgage_pd_suite.joblib")
    if path.exists():
        try:
            from ifrs9_cockpit.models.pd_model import PDModelSuite
            return PDModelSuite.load(str(path))
        except Exception:
            return None
    return None


@functools.lru_cache(maxsize=1)
def load_governance_artifacts() -> Optional[Dict[str, Any]]:
    """Charge les artefacts de gouvernance pre-calcules.

    Cherche governance_suite.joblib produit par le pipeline offline
    (train.py --output-governance). Si absent, retourne None et le
    dashboard calcule inline comme avant (zero regression).

    Returns:
        Dict d'artefacts ou None si le fichier n'existe pas.
    """
    path = Path("ifrs9_cockpit/training/models/governance_suite.joblib")
    if path.exists():
        try:
            import joblib
            return joblib.load(str(path))
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────
# GPU SAFETY (force CPU for CUDA nightly issues)
# ──────────────────────────────────────────────
def force_models_cpu(pd_suite) -> None:
    """Force tous les modeles sur CPU pour eviter segfault CUDA nightly Blackwell.

    Le nightly PyTorch 2.11.0+cu128 pour l'architecture sm_120 (RTX 5070 Ti)
    peut provoquer des segfault lors de l'inference TabNet apres serialisation.
    Cette fonction deplace tous les tenseurs sur CPU apres chargement.

    Args:
        pd_suite: PDModelSuite chargee (potentiellement avec tenseurs GPU).
    """
    try:
        import torch as _torch
    except ImportError:
        return
    if not _torch.cuda.is_available():
        return

    _cpu = _torch.device("cpu")

    def _move_all_tensors(module: _torch.nn.Module) -> None:
        """Deplace recursivement tous les tenseurs d'un module sur CPU."""
        module.to(_cpu)
        for m in module.modules():
            for attr_name in list(m.__dict__.keys()):
                attr = m.__dict__[attr_name]
                if isinstance(attr, _torch.Tensor) and attr.device != _cpu:
                    m.__dict__[attr_name] = attr.to(_cpu)

    for _name, result in pd_suite.results.items():
        model = result.model
        if not hasattr(model, "calibrated_classifiers_"):
            continue
        for cc in model.calibrated_classifiers_:
            inner = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if inner is None:
                continue
            # TabNet : deplacer le reseau PyTorch
            if hasattr(inner, "_model") and hasattr(inner._model, "network"):
                try:
                    _move_all_tensors(inner._model.network)
                    inner._model.device_name = "cpu"
                    inner._model.device = _cpu
                except Exception:
                    pass
            # XGBoost : forcer device=cpu
            if hasattr(inner, "set_params"):
                try:
                    inner.set_params(device="cpu")
                except (TypeError, ValueError, AttributeError):
                    pass


# ──────────────────────────────────────────────
# SHAP VALUES CACHE
# ──────────────────────────────────────────────
_shap_cache: Dict[str, np.ndarray] = {}
_SHAP_MAX_SAMPLES = 500
_SHAP_BACKGROUND = 50


def compute_shap_values(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str,
) -> np.ndarray:
    """Calcule et cache les SHAP values (cle = model_name + n_samples).

    Strategie d'explainer par famille de modele :
        - LR_WoE   : LinearExplainer (analytique, rapide)
        - XGBoost   : TreeExplainer (path-based, exact)
        - TabNet    : KernelExplainer (model-agnostic, plus lent)

    Args:
        model: Modele sklearn-compatible avec predict_proba.
        X_sample: Matrice de features (n_samples, n_features).
        feature_names: Noms des features (pour reference).
        model_name: Nom du modele (LR_WoE, TabNet, XGBoost).

    Returns:
        Array SHAP values (n_samples, n_features) pour la classe positive.
    """
    cache_key = f"{model_name}_{X_sample.shape[0]}"
    if cache_key in _shap_cache:
        return _shap_cache[cache_key]

    import shap

    if X_sample.shape[0] > _SHAP_MAX_SAMPLES:
        X_sample = X_sample[:_SHAP_MAX_SAMPLES]

    if model_name == "LR_WoE":
        explainer = shap.LinearExplainer(model, X_sample)
    elif model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
    else:
        # TabNet ou tout autre modele : KernelExplainer (model-agnostic)
        n_bg = min(_SHAP_BACKGROUND, X_sample.shape[0])
        background = shap.sample(X_sample, n_bg)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )

    shap_vals = explainer.shap_values(X_sample)

    # Normaliser la sortie : certains explainers retournent une liste [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    _shap_cache[cache_key] = shap_vals
    return shap_vals
