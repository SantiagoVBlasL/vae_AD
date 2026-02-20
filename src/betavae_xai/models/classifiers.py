"""
models/classifiers.py

Classical ML classifiers (scikit-learn, LightGBM, XGBoost) and
their Optuna search spaces for AD vs CN classification from VAE latent
representations.

Used in:
"Explainable Latent Representation Learning for Alzheimer’s Disease:
 A β-VAE and Saliency Map Framework"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from lightgbm import LGBMClassifier
from lightgbm.basic import _LIB, _safe_call
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier


logger = logging.getLogger(__name__)

# Silenciar un poco librerías ruidosas
for noisy in ["lightgbm", "optuna", "sklearn", "xgboost"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# LightGBM C++ log level (si la función existe)
if hasattr(_LIB, "LGBM_SetLogLevel"):
    try:
        # 0 = fatal, 1 = error, 2 = warning, 3 = info, 4 = debug
        _safe_call(_LIB.LGBM_SetLogLevel(0))
    except Exception:
        pass

# XGBoost: esconder logs molestos
os.environ.setdefault("XGB_HIDE_LOG", "1")

# Detección de GPU opcional (cupy)
try:
    import cupy as cp

    HAS_GPU: bool = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_GPU = False


ClassifierPipelineAndGrid = Tuple[ImblearnPipeline, Dict[str, Any], int]

def _sex_to_float(x):
    """
    Robust encoder for Sex:
      - strings: M/m/male -> 0, F/f/female -> 1
      - numeric: 0/1 kept
      - unknown -> NaN (to be imputed)
    Accepts (n, 1) or (n,) array-like. Returns (n, 1) float array.
    """
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    s = pd.Series(arr)

    # If already numeric (or numeric strings), coerce first
    s_num = pd.to_numeric(s, errors="coerce")
    out = s_num.copy()

    # For non-numeric entries, map common strings
    mask_nan = out.isna()
    if mask_nan.any():
        s_str = s[mask_nan].astype(str).str.strip().str.lower()
        mapping = {
            "m": 0.0, "male": 0.0, "masc": 0.0, "masculino": 0.0,
            "f": 1.0, "female": 1.0, "fem": 1.0, "femenino": 1.0,
        }
        out.loc[mask_nan] = s_str.map(mapping)

    return out.to_numpy(dtype=float).reshape(-1, 1)


def _to_numeric_df(df: pd.DataFrame) -> np.ndarray:
    """Coerce a DataFrame to numeric (errors -> NaN), return 2D numpy array."""
    return df.apply(pd.to_numeric, errors="coerce").to_numpy()


class _AutoPreprocessor(BaseEstimator, TransformerMixin):
    """
    Single preprocessing step:
      - DataFrame input: encode Sex safely; coerce numeric columns; impute; (optionally) scale numeric.
      - Numpy input: impute; (optionally) scale numeric.
    """
    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric

    def fit(self, X, y=None):
        self._is_df_ = isinstance(X, pd.DataFrame)

        if self._is_df_:
            cols = list(X.columns)
            sex_cols = [c for c in cols if str(c).strip().lower() == "sex"]
            other_cols = [c for c in cols if c not in sex_cols]

            # Keep only columns that have at least one numeric value after coercion (avoid all-NaN columns)
            num_cols: List[str] = []
            for c in other_cols:
                s = pd.to_numeric(X[c], errors="coerce")
                if s.notna().any():
                    num_cols.append(c)

            transformers: List[tuple] = []

            if sex_cols:
                sex_steps = [
                    ("sex_enc", FunctionTransformer(_sex_to_float, validate=False)),
                   ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
                if self.scale_numeric:
                    sex_steps.append(("scaler", StandardScaler()))
                transformers.append(("sex", Pipeline(sex_steps), sex_cols))

            if num_cols:
                num_steps = [
                   ("to_num", FunctionTransformer(_to_numeric_df, validate=False)),
                    ("imputer", SimpleImputer(strategy="median")),
                ]
                if self.scale_numeric:
                    num_steps.append(("scaler", StandardScaler()))
                transformers.append(("num", Pipeline(num_steps), num_cols))

            self._ct_ = ColumnTransformer(transformers=transformers, remainder="drop")
            self._ct_.fit(X, y)
        else:
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.scale_numeric:
                steps.append(("scaler", StandardScaler()))
            self._ct_ = Pipeline(steps)
            self._ct_.fit(X, y)

        return self

    def transform(self, X):
        return self._ct_.transform(X)



def get_available_classifiers() -> List[str]:
    """Devuelve la lista de tipos de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp", "xgb"]


def _parse_hidden_layers(hidden_layers_str: str | None) -> Tuple[int, ...]:
    """Convierte un string '128,64' en una tupla (128, 64)."""
    if not hidden_layers_str:
        return (128, 64)
    return tuple(
        int(x.strip())
        for x in hidden_layers_str.split(",")
        if x.strip()
    )


def get_classifier_and_grid(
    classifier_type: str,
    *,
    seed: int = 42,
    balance: bool = False,
    use_smote: bool = False,
    tune_sampler_params: bool = False,
    mlp_hidden_layers: str = "128,64",
    calibrate: bool = False,
    use_feature_selection: bool = False,
) -> ClassifierPipelineAndGrid:
    """
    Construye un pipeline de imblearn y devuelve:
      - pipeline (ImblearnPipeline)
      - dict de distribuciones de Optuna
      - n_iter_suggested (int) para la búsqueda.

    Los features de entrada típicos son:
      - vectores latentes del VAE (z, mu, etc.)
      - opcionalmente, metadatos concatenados.
    """
    ctype = classifier_type.lower()
    if ctype not in get_available_classifiers():
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type!r}")

    class_weight = "balanced" if balance else None
    model: Any
    param_distributions: Dict[str, Any]
    n_iter_search = 150  # valor por defecto

    # ------------------------------------------------------------------
    # SVM (RBF)
    # ------------------------------------------------------------------
    if ctype == "svm":
        model = SVC(
            probability=False,
            random_state=seed,
            class_weight=class_weight,
            cache_size=10000,
        )
        param_distributions = {
            "model__C": FloatDistribution(1e-1, 1e4, log=True),
            "model__gamma": FloatDistribution(1e-7, 1e-1, log=True),
            "model__kernel": CategoricalDistribution(["rbf"]),
        }
        n_iter_search = 900

    # ------------------------------------------------------------------
    # Regresión logística
    # ------------------------------------------------------------------
    elif ctype == "logreg":
        model = LogisticRegression(
            random_state=seed,
            class_weight=class_weight,
            solver="liblinear",
            max_iter=20000,
        )
        param_distributions = {
            "model__C": FloatDistribution(1e-5, 1, log=True),
        }
        n_iter_search = 900

    # ------------------------------------------------------------------
    # Gradient Boosting con LightGBM
    # ------------------------------------------------------------------
    elif ctype == "gb":
        model = LGBMClassifier(
            random_state=seed,
            objective="binary",
            class_weight=class_weight,
            n_jobs=1,   # Optuna se encarga de la paralelización externa
            verbose=-1,
        )

        # Soporte GPU de LightGBM (si el build lo permite)
        if HAS_GPU:
            try:
                # LGBM_HasGPU devuelve 0/1
                if bool(_safe_call(_LIB.LGBM_HasGPU())):
                    model.set_params(device_type="gpu", gpu_use_dp=True)
                    logger.info("[LightGBM] ➜ GPU activada")
                else:
                    model.set_params(device_type="cpu")
                    logger.info("[LightGBM] ➜ Build sin GPU, usando CPU")
            except Exception:
                model.set_params(device_type="cpu")
                logger.info("[LightGBM] ➜ No se pudo comprobar la GPU, usando CPU")
        else:
            model.set_params(device_type="cpu")
            logger.info("[LightGBM] ➜ GPU no disponible, usando CPU")

        param_distributions = {
            # Estructura del árbol
            "model__max_depth":        IntDistribution(3, 12),
            "model__num_leaves":       IntDistribution(4, 64),

            # Muestras y features por árbol
            "model__bagging_fraction": FloatDistribution(0.5, 1.0),
            "model__feature_fraction": FloatDistribution(0.5, 1.0),
            "model__bagging_freq":     IntDistribution(1, 10),

            # Aprendizaje
            "model__learning_rate":    FloatDistribution(5e-4, 0.01, log=True),
            "model__n_estimators":     IntDistribution(300, 1000),

            # Regularización
            "model__min_child_samples": IntDistribution(5, 50),
            "model__min_child_weight":  FloatDistribution(1e-3, 10, log=True),
            "model__min_split_gain":    FloatDistribution(0.0, 1.0),
            "model__reg_alpha":         FloatDistribution(1e-3, 1.0, log=True),
            "model__reg_lambda":        FloatDistribution(1e-3, 1.0, log=True),
        }

        n_param = len(param_distributions)
        n_iter_search = int(round((15 * n_param) / 10.0)) * 10  # múltiplo de 10

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    elif ctype == "rf":
        logger.info("[RandomForest] ➜ scikit-learn (CPU).")
        model = RandomForestClassifier(
            random_state=seed,
            class_weight=class_weight,
            n_jobs=-1,
        )
        param_distributions = {
            "model__n_estimators":      IntDistribution(100, 1200),
            "model__max_features":      CategoricalDistribution(["sqrt", "log2", 0.2, 0.4]),
            "model__max_depth":         IntDistribution(8, 50),
            "model__min_samples_split": IntDistribution(2, 30),
            "model__min_samples_leaf":  IntDistribution(1, 20),
        }
        n_iter_search = 150

    # ------------------------------------------------------------------
    # MLP (scikit-learn)
    # ------------------------------------------------------------------
    elif ctype == "mlp":
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(
            random_state=seed,
            hidden_layer_sizes=hidden,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=25,
        )
        param_distributions = {
            "model__alpha":             FloatDistribution(1e-5, 1e-1, log=True),
            "model__learning_rate_init":FloatDistribution(1e-5, 1e-2, log=True),
        }
        n_iter_search = 200

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    elif ctype == "xgb":
        device = "cuda" if HAS_GPU else "cpu"
        model = XGBClassifier(
            random_state=seed,
            eval_metric="auc",
            n_jobs=1,
            tree_method="hist",
            device=device,
            verbosity=0,
        )
        if HAS_GPU:
            logger.info("[XGBoost] ➜ se usará GPU (device=cuda)")
        else:
            logger.info("[XGBoost] ➜ GPU no disponible, usando CPU.")

        param_distributions = {
            "model__gamma":            FloatDistribution(0.0, 5.0),
            "model__n_estimators":     IntDistribution(500, 1500),
            "model__learning_rate":    FloatDistribution(1e-4, 0.1, log=True),
            "model__max_depth":        IntDistribution(2, 8),
            "model__subsample":        FloatDistribution(0.3, 1.0),
            "model__colsample_bytree": FloatDistribution(0.5, 1.0),
            "model__min_child_weight": FloatDistribution(0.5, 10.0, log=True),
        }
        n_iter_search = 200


    # ------------------------------------------------------------------
    # Calibración (DEPRECADO aquí)
    # ------------------------------------------------------------------
    if calibrate:
        logger.warning(
            "[classifiers.py] calibrate=True está DEPRECADO. "
            "Recomendación: calibrar post-hoc *después* del tuning, "
            "envolviendo el pipeline ya fitteado (mejor para SHAP y evita renombrar params)."
        )

    # ------------------------------------------------------------------
    # Construcción del pipeline Imblearn
    # ------------------------------------------------------------------
    # 1) Preprocesado: encode Sex + impute (+ optional scale) in a single safe step
    tree_models = {"rf", "gb", "xgb"}  # modelos basados en árboles no necesitan escalado, pero SVM y MLP sí.
    # SMOTE usa distancias → si está activo, conviene escalar aunque el modelo final sea de árbol.
    scale_numeric = True if use_smote else (ctype not in tree_models)
    preprocess_step = ("preprocess", _AutoPreprocessor(scale_numeric=scale_numeric))
 

    # 2) SMOTE opcional
    oversampler_step: tuple | None = None
    if use_smote:
        oversampler_step = ("smote", SMOTE(random_state=seed))
        logger.info("[SMOTE] ➜ aplicado sólo dentro de folds (imblearn Pipeline).")
        if tune_sampler_params:
            param_distributions["smote__k_neighbors"] = IntDistribution(3, 25)

    # 3) Selección de features opcional (para latentes + metadatos)
    feature_selector_step: tuple | None = None
    if use_feature_selection:
        feature_selector_step = ("feature_selector", SelectKBest(f_classif))
        param_distributions["feature_selector__k"] = IntDistribution(20, 256)
        logger.info("[SelectKBest] ➜ añadido al pipeline (k tunable 20–256).")

    # 4) Clasificador
    model_step = ("model", model)

    # 5) Ensamblar pasos en orden lógico
    steps: List[tuple] = [preprocess_step]
    # Orden recomendado:
    # preprocess → (feature selection) → (SMOTE) → model
    # Así evitamos que puntos sintéticos influyan en la selección de features.
    if feature_selector_step is not None:
        steps.append(feature_selector_step)
    if oversampler_step is not None:
        steps.append(oversampler_step)
    steps.append(model_step)

    full_pipeline = ImblearnPipeline(steps=steps)
    return full_pipeline, param_distributions, n_iter_search


__all__ = [
    "ClassifierPipelineAndGrid",
    "get_available_classifiers",
    "get_classifier_and_grid",
]