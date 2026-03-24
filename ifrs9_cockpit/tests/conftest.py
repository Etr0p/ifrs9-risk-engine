import os

# Limiter les threads BLAS/OpenMP AVANT tout import numpy/scipy.
# Evite la surcharge N_workers * N_blas_threads sur un CPU multi-core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pytest
import numpy as np
import json
import joblib
from pathlib import Path
from filelock import FileLock

# Smart test selector — --smart, --all, --slow flags + git-based selection
from ifrs9_cockpit.tests.smart_test_selector import (  # noqa: F401
    pytest_addoption,
    pytest_collection_modifyitems,
)

from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.utils.frame_compat import to_pandas
from ifrs9_cockpit.config import RANDOM_SEED

# Chemin du modele PD pre-entraine (1.5M clients via train.py)
_PRETRAINED_PD = Path(__file__).resolve().parent.parent / "training" / "models" / "pd_suite.joblib"

@pytest.fixture(scope="session")
def global_pipeline_results(tmp_path_factory, worker_id):
    """Calcule et partage les résultats du pipeline entre les workers xdist."""

    # Répertoire de partage pour la session
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    fn = root_tmp_dir / "pipeline_data.joblib"

    # Lock pour éviter que 32 workers calculent en même temps
    with FileLock(str(fn) + ".lock"):
        if fn.exists():
            return joblib.load(fn)

        # Calcul (exécuté par UN SEUL worker)
        df_credit, df_pe, df_history, _ = generate_dataset(n_clients=1000, seed=RANDOM_SEED)

        # Charger le modele PD pre-entraine (1.5M) au lieu de re-entrainer
        # pd_model/lgd_model/ead_model still use Pandas internally
        df_credit_pd = to_pandas(df_credit)
        if _PRETRAINED_PD.exists():
            pd_suite = PDModelSuite.load(str(_PRETRAINED_PD))
        else:
            pd_suite = PDModelSuite(seed=RANDOM_SEED)
            pd_suite.fit(df_credit_pd)
        pd_current = pd_suite.predict_active(df_credit_pd)
        pd_origination = df_credit["pd_origination"].to_numpy() if hasattr(df_credit["pd_origination"], "to_numpy") else df_credit["pd_origination"].values

        lgd_model = LGDModel()
        lgd_model.fit(df_credit_pd)

        ead_model = EADModel()
        ead_model.fit(df_credit_pd)

        ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
        result_credit = ecl_calc.calculate(df_credit, pd_current, pd_origination)

        pe_calc = PECalculator()
        result_pe = pe_calc.calculate(df_pe)

        comparator = PortfolioComparator(result_credit, result_pe)

        data = {
            "df_credit": df_credit,
            "df_pe": df_pe,
            "df_history": df_history,
            "pd_current": pd_current,
            "pd_origination": pd_origination,
            "result_credit": result_credit,
            "result_pe": result_pe,
            "comparator": comparator,
            "pd_suite": pd_suite,
            "lgd_model": lgd_model,
            "ead_model": ead_model
        }

        joblib.dump(data, fn)
        return data

@pytest.fixture(scope="session")
def global_dataset(global_pipeline_results):
    return (
        global_pipeline_results["df_credit"],
        global_pipeline_results["df_pe"],
        global_pipeline_results["df_history"]
    )

@pytest.fixture(scope="session")
def global_trained_pd_suite(global_pipeline_results):
    return global_pipeline_results["pd_suite"]

@pytest.fixture(scope="session")
def global_lgd_model(global_pipeline_results):
    return global_pipeline_results["lgd_model"]

@pytest.fixture(scope="session")
def global_ead_model(global_pipeline_results):
    return global_pipeline_results["ead_model"]

@pytest.fixture(scope="session")
def global_result_credit(global_pipeline_results):
    return global_pipeline_results["result_credit"]

@pytest.fixture(scope="session")
def global_result_pe(global_pipeline_results):
    return global_pipeline_results["result_pe"]

@pytest.fixture(scope="session")
def global_comparator(global_pipeline_results):
    return global_pipeline_results["comparator"]

@pytest.fixture(scope="session")
def global_df_pe(global_pipeline_results):
    return global_pipeline_results["df_pe"]
