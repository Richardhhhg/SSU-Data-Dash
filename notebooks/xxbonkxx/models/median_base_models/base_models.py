"""Base models trained on a target scaled by one categorical feature.

For a scaling feature (say house_style) the target is log(sold_price) minus the median
offset of that house_style, and the model trains on every other feature. Predictions get
the offset added back, so every base model predicts log sold price and can be stacked.

To use another library, write a class with the same methods as XGBModel and add it to MODELS.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from notebooks.xxbonkxx.models.xxbonkxx_features import (
    apply_scaling_offset,
    fit_scaling_offset,
    off_market_features,
)

SAVE_DIR = Path(__file__).parent / "base_preds"


class XGBModel:
    def __init__(
        self,
        params,
        device="cpu",
        seed=None,
        n_estimators=2000,
        early_stopping_rounds=100,
    ):
        self.params = params
        self.device = device
        self.seed = seed
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        """Early stopping on (X_va, y_va) if given, otherwise exactly n_estimators rounds."""
        early_stop = X_va is not None
        self.model = XGBRegressor(
            **self.params,
            enable_categorical=True,
            device=self.device,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            early_stopping_rounds=self.early_stopping_rounds if early_stop else None,
        )
        eval_set = [(X_va, y_va)] if early_stop else None
        self.model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def n_rounds(self):
        """Boosting rounds actually used (the best iteration when early stopping)."""
        best = getattr(self.model, "best_iteration", None)
        return self.n_estimators if best is None else best + 1

    @property
    def feature_importances(self):
        return pd.Series(
            self.model.feature_importances_, index=self.model.feature_names_in_
        )

    @staticmethod
    def suggest_params(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "max_depth": trial.suggest_int("max_depth", 3, 13),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.1, 1.0, step=0.1
            ),
            "max_bin": trial.suggest_int("max_bin", 256, 1024),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.1, step=0.01),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        }


MODELS = {"xgb": XGBModel}


@dataclass
class BaseModelConfig:
    name: str  # e.g. "style"; used for the saved files and prediction column
    group_col: str  # column in listings to scale by, e.g. "house_style"
    drop_cols: list[str]  # columns to drop from X, e.g. ["house_style"]
    make_features: Callable = off_market_features
    model: str = "xgb"
    n_trials: int = 20
    k_folds: int = 5
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    debug: bool = False  # print MdAPE and MAPE for every fold

    def build_model(self, params, device, seed):
        return MODELS[self.model](
            params,
            device=device,
            seed=seed,
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def build_data(self, homes):
        """X without the scaling column(s), log sold price, and the raw scaling column."""
        X = self.make_features(homes).drop(columns=self.drop_cols)
        y_log = np.log(homes["sold_price"])
        return X, y_log, homes[self.group_col]


def align_categories(X_tr, X_va):
    """Give X_va's categorical columns the categories of X_tr. Unseen values become missing."""
    X_va = X_va.copy()
    for col in X_tr.select_dtypes("category"):
        X_va[col] = X_va[col].cat.set_categories(X_tr[col].cat.categories)
    return X_va


def percent_errors(y_log, pred_log):
    return 100 * np.abs(np.exp(np.asarray(pred_log)) / np.exp(np.asarray(y_log)) - 1)


def cross_val_oof(cfg, X, y_log, groups, params, device="cpu", seed=None):
    """Walk-forward K-fold on rows sorted by time. In each fold the offset is fit on the
    training part only. Returns (mean MdAPE, out-of-fold log price predictions, mean
    boosting rounds). The OOF array is NaN for the first block, which is never validated."""
    oof = np.full(len(X), np.nan)
    mdapes, rounds = [], []
    for fold, (tr, va) in enumerate(
        TimeSeriesSplit(n_splits=cfg.k_folds).split(X), start=1
    ):
        offset = fit_scaling_offset(groups.iloc[tr], y_log.iloc[tr])
        off_tr = apply_scaling_offset(groups.iloc[tr], offset)
        off_va = apply_scaling_offset(groups.iloc[va], offset)

        model = cfg.build_model(params, device, seed)
        model.fit(
            X.iloc[tr], y_log.iloc[tr] - off_tr, X.iloc[va], y_log.iloc[va] - off_va
        )
        oof[va] = model.predict(X.iloc[va]) + off_va

        error = percent_errors(y_log.iloc[va], oof[va])
        mdapes.append(np.median(error))
        rounds.append(model.n_rounds)
        if cfg.debug:
            print(
                f"[{cfg.name}] fold {fold}: train {len(tr):,}  val {len(va):,}  "
                f"MdAPE {np.median(error):.2f}%  MAPE {error.mean():.2f}%",
                flush=True,
            )

    print(f"[{cfg.name}] mean MdAPE {np.mean(mdapes):.2f}%", flush=True)
    return np.mean(mdapes), oof, int(np.mean(rounds))


def tune(cfg, X, y_log, groups, device="cpu", seed=None):
    """Optuna search over the model's suggest_params. Returns the study."""
    suggest = MODELS[cfg.model].suggest_params

    def objective(trial):
        return cross_val_oof(cfg, X, y_log, groups, suggest(trial), device, seed)[0]

    study = optuna.create_study(
        direction="minimize",
        study_name=cfg.name,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=cfg.n_trials)
    return study


def run_base_model(
    cfg: BaseModelConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    device="cpu",
    seed=None,
    params=None,
    save=True,
):
    """Tune (unless params are given), get OOF predictions on train, then refit on all of
    train and predict val. Predictions are log sold prices. With save=True they go to
    base_preds/{name}_train.parquet and base_preds/{name}_val.parquet."""
    X_tr, y_tr, g_tr = cfg.build_data(train)
    X_va, y_va, g_va = cfg.build_data(val)
    X_va = align_categories(X_tr, X_va)

    study = None
    if params is None:
        study = tune(cfg, X_tr, y_tr, g_tr, device, seed)
        params = study.best_params

    _, oof, n_rounds = cross_val_oof(cfg, X_tr, y_tr, g_tr, params, device, seed)

    offset = fit_scaling_offset(g_tr, y_tr)
    model = cfg.build_model(params, device, seed)
    model.n_estimators = n_rounds
    model.fit(X_tr, y_tr - apply_scaling_offset(g_tr, offset))
    pred_val = model.predict(X_va) + apply_scaling_offset(g_va, offset)

    oof_train = pd.Series(oof, index=train.index, name=cfg.name)
    pred_val = pd.Series(pred_val, index=val.index, name=cfg.name)
    if save:
        SAVE_DIR.mkdir(exist_ok=True)
        oof_train.to_frame().to_parquet(SAVE_DIR / f"{cfg.name}_train.parquet")
        pred_val.to_frame().to_parquet(SAVE_DIR / f"{cfg.name}_val.parquet")

    error = percent_errors(y_va, pred_val)
    print(
        f"[{cfg.name}] val  MdAPE {np.median(error):.2f}%  MAPE {error.mean():.2f}%",
        flush=True,
    )
    return {
        "oof_train": oof_train,
        "pred_val": pred_val,
        "best_params": params,
        "model": model,
        "offset": offset,
        "study": study,
    }
