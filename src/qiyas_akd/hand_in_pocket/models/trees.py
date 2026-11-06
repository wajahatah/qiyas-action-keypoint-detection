"""Tree ensembles on flattened windows (approaches A-C): Random Forest, XGBoost, LightGBM.

Defaults are the values the original experiments used:

* RF — 100 trees, sklearn defaults otherwise. Its grid search (:data:`RF_PARAM_GRID`) is preserved,
  with one bug fixed: ``max_depth`` listed the *string* ``'None'`` (invalid for sklearn) instead of
  ``None``.
* XGBoost — 100 trees, depth 3, learning rate 0.1, logloss. The GPU arguments of the original are
  dropped (``tree_method="hist"`` runs on CPU and GPU; pass ``device="cuda"`` via params for GPU).
* LightGBM — 100 trees, 31 leaves, learning rate 0.7 (sic — aggressive, and the model with the most
  false positives on the bench), unlimited depth.

``xgboost`` and ``lightgbm`` are optional extras (``pip install 'qiyas-akd[boosting]'``) and are
only imported when such a model is requested.

Grid search uses ``GroupKFold`` by video instead of the original random 5-fold CV, for the same
leakage reason as ``evaluation.grouped_split``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

TREE_KINDS = ("rf", "xgboost", "lightgbm")

RF_PARAM_GRID: dict[str, list[Any]] = {
    "max_depth": [None, 5, 7, 10, 13],
    "n_estimators": [75, 100, 150, 200],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy", "log_loss"],
}


def make_tree_model(kind: str, params: Mapping[str, Any] | None = None, seed: int = 42) -> Any:
    """Unfitted sklearn-API classifier with the original defaults, overridable by ``params``."""
    p = dict(params or {})
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**{"n_estimators": 100, "random_state": seed, **p})
    if kind == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("XGBoost model needs: pip install 'qiyas-akd[boosting]'") from exc
        defaults = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": seed,
        }
        return xgb.XGBClassifier(**{**defaults, **p})
    if kind == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("LightGBM model needs: pip install 'qiyas-akd[boosting]'") from exc
        defaults = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.7,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": seed,
            "verbose": -1,
        }
        return lgb.LGBMClassifier(**{**defaults, **p})
    raise ValueError(f"unknown tree model {kind!r}; choose from {TREE_KINDS}")


def grid_search(
    model: Any,
    grid: Mapping[str, Sequence[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: int = 5,
    scoring: str = "f1",
) -> Any:
    """Fit ``GridSearchCV`` with video-grouped folds and return the refitted best estimator.

    The original scored by accuracy, which on ~90 % negatives mostly rewards predicting "no";
    F1 on the positive class is the default here.
    """
    from sklearn.model_selection import GridSearchCV, GroupKFold

    n_groups = len(np.unique(groups))
    search = GridSearchCV(
        model, dict(grid), cv=GroupKFold(n_splits=min(cv, n_groups)), scoring=scoring, n_jobs=1
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_
