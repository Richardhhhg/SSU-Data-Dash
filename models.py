"""The models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track: a meta model stacking four gradient-boosted
tree models (XGBoost, CatBoost, LightGBM, and LightGBM with one model for apartments and
one for houses). The two baselines stay for comparison.

The tree models' hyperparameters were tuned with optuna in notebooks/xxbonkxx/models/
(copied from best_params.py there).
"""

import hashlib
import time
from pathlib import Path

import catboost
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

import data
import features

SEED = 987
USE_GPU = (
    get_gpu_device_count() > 0
)  # XGBoost and CatBoost train on the GPU when there is one

XGBOOST_PARAMS = {
    "learning_rate": 0.06999999999999999,
    "max_depth": 8,
    "n_estimators": 1600,
    "subsample": 0.9,
    "colsample_bytree": 0.4,
    "max_bin": 297,
    "min_child_weight": 10,
    "gamma": 0.0,
    "reg_lambda": 0.2473645458361432,
    "reg_alpha": 1.2584559777650532,
    "max_delta_step": 7,
}

CATBOOST_PARAMS = {
    "learning_rate": 0.09999999999999999,
    "depth": 9,
    "iterations": 1400,
    "l2_leaf_reg": 0.040807415431426686,
    "random_strength": 2.5265354019008015,
    "subsample": 1.0,
    "border_count": 254,
    "one_hot_max_size": 7,
}

LIGHTGBM_PARAMS = {
    "learning_rate": 0.08,
    "num_leaves": 160,
    "max_depth": 12,
    "n_estimators": 1000,
    "min_child_samples": 43,
    "min_split_gain": 0.0,
    "reg_lambda": 0.00691982217304558,
    "reg_alpha": 0.0015826257024873476,
    "feature_fraction": 0.5,
}

LIGHTGBM_APARTMENT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 195,
    "max_depth": 6,
    "n_estimators": 1800,
    "colsample_bytree": 0.4,
    "min_child_samples": 68,
    "min_split_gain": 0.0,
    "reg_lambda": 0.0017675392438743578,
    "reg_alpha": 1.5005788903973847,
}

LIGHTGBM_HOUSE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 248,
    "max_depth": 10,
    "n_estimators": 1000,
    "colsample_bytree": 0.6,
    "min_child_samples": 13,
    "min_split_gain": 0.0,
    "reg_lambda": 1.482210462847529,
    "reg_alpha": 0.5155994576469242,
}

AWARE_XGBOOST_PARAMS = {
    "learning_rate": 0.02,
    "max_depth": 13,
    "n_estimators": 1200,
    "subsample": 0.7000000000000001,
    "colsample_bytree": 0.9,
    "max_bin": 1523,
    "min_child_weight": 5,
    "gamma": 0.01,
    "reg_lambda": 0.09067060845188568,
    "reg_alpha": 2.1221007442196798,
    "max_delta_step": 7,
}

AWARE_CATBOOST_PARAMS = {
    "learning_rate": 0.05,
    "depth": 10,
    "iterations": 2000,
    "l2_leaf_reg": 4.809832193856093,
    "random_strength": 3.0009335320617767,
    "subsample": 0.9,
    "border_count": 254,
    "one_hot_max_size": 9,
}

AWARE_LIGHTGBM_PARAMS = {
    "learning_rate": 0.04,
    "num_leaves": 235,
    "max_depth": 6,
    "n_estimators": 2000,
    "colsample_bytree": 0.30000000000000004,
    "min_child_samples": 22,
    "min_split_gain": 0.0,
    "reg_lambda": 0.824461364065615,
    "reg_alpha": 0.005510621165738579,
}

AWARE_LIGHTGBM_APARTMENT_PARAMS = {
    "learning_rate": 0.060000000000000005,
    "num_leaves": 49,
    "max_depth": 13,
    "n_estimators": 1000,
    "colsample_bytree": 0.8,
    "min_child_samples": 82,
    "min_split_gain": 0.0,
    "reg_lambda": 0.6933877086258585,
    "reg_alpha": 0.01197099282824542,
}

AWARE_LIGHTGBM_HOUSE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 212,
    "max_depth": 6,
    "n_estimators": 1800,
    "colsample_bytree": 0.5,
    "min_child_samples": 25,
    "min_split_gain": 0.0,
    "reg_lambda": 0.05107704444922434,
    "reg_alpha": 2.8015668038585706,
}

META_MONTHS = 6  # months of out-of-fold predictions the meta model learns from
META_ALPHA = 1.931
AWARE_META_ALPHA = 0.3728

TRACK_PARAMS = {
    "off_market": {
        "xgboost": XGBOOST_PARAMS,
        "catboost": CATBOOST_PARAMS,
        "lightgbm": LIGHTGBM_PARAMS,
        "lightgbm_split_apartment": {
            1: LIGHTGBM_APARTMENT_PARAMS,
            0: LIGHTGBM_HOUSE_PARAMS,
        },
        "meta_alpha": META_ALPHA,
    },
    "list_aware": {
        "xgboost": AWARE_XGBOOST_PARAMS,
        "catboost": AWARE_CATBOOST_PARAMS,
        "lightgbm": AWARE_LIGHTGBM_PARAMS,
        "lightgbm_split_apartment": {
            1: AWARE_LIGHTGBM_APARTMENT_PARAMS,
            0: AWARE_LIGHTGBM_HOUSE_PARAMS,
        },
        "meta_alpha": AWARE_META_ALPHA,
    },
}

VERBOSE = True

# Base-model predictions are cached in memory and, when CACHE_DIR is set, on disk.
# Walk-forward validation asks for the same ones many times (month M's final models
# are month M+1's out-of-fold models), and predict reuses validate's out-of-fold months.
# A cache entry is keyed by the code, the library versions and the data, so editing
# models.py, features.py or data.py starts afresh. Delete the folder to clear it.
CACHE_DIR = None
_BASE_LOG_PREDICTIONS = {}


def log(message):
    if VERBOSE:
        print(message, flush=True)


def _fingerprint(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else str(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def _table_fingerprint(df):
    """Changes whenever any value, row order or column of `df` changes."""
    rows = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return _fingerprint(rows.tobytes(), list(df.columns))


def _code_fingerprint():
    folder = Path(__file__).parent
    sources = [(folder / name).read_bytes() for name in ("models.py", "features.py", "data.py")]
    versions = [m.__version__ for m in (np, pd, sklearn, xgboost, lightgbm, catboost)]
    return _fingerprint(*sources, *versions)


CODE_FINGERPRINT = _code_fingerprint()


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    # return ListPriceBaseline() if track == "list_aware" else NearestSalesBaseline(k=10)
    return MetaModel(track)


def fit_or_load(track, train, path):
    """make_model(track).fit(train), saved to `path` and loaded from there next time.

    The saved model holds the meta model and the final base models that predict uses.
    Whenever `path` exists it is loaded as is, without checking that it was trained on
    this `train` or with this code: delete it after changing either, to train again.
    """
    path = Path(path)
    if path.exists():
        model = joblib.load(path)["model"]
        if isinstance(model, MetaModel):
            model.attach_train(train)
        log(f"Loaded {track} model from {path}")
        return model
    model = make_model(track).fit(train)
    if isinstance(model, MetaModel):
        model.fit_final_base_models()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, path)
    log(f"Saved {track} model to {path}")
    return model


def make_features_for(track):
    """The feature function of a track: list-aware homes also have their list price."""
    return (
        features.list_aware_features
        if track == "list_aware"
        else features.off_market_features
    )


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self  # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = [
            "ml_num",
            "house_type_name",
            "date_sold",
            "latitude",
            "longitude",
            "sold_price",
        ]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(
            ["date_sold", "ml_num"], ascending=[False, True]
        )
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:  # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx**2 + dy**2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class TreeModel:
    def __init__(self, make_features, params):
        self.make_features = make_features
        self.params = params

    def fit(self, train):
        X, y = self.training_features(train)
        self.regressor = self.new_regressor().fit(X, y)
        return self

    def predict(self, homes):
        return np.exp(self.regressor.predict(self.features_to_value(homes)))

    def __getstate__(self):
        # The training table is not saved with the model: MetaModel.attach_train puts it back.
        return {name: value for name, value in self.__dict__.items() if name != "train"}

    def training_features(self, train):
        self.train = train
        start = time.perf_counter()
        X = self.make_features(train)
        self.feature_seconds = time.perf_counter() - start
        self.categories = {
            column: X[column].cat.categories for column in X.select_dtypes("category")
        }
        return self.set_categorical_columns(X), np.log(train["sold_price"])

    def features_to_value(self, homes):
        return self.set_categorical_columns(
            self.make_features(homes, history=self.train)
        )

    def set_categorical_columns(self, X):
        for column, categories in self.categories.items():
            X[column] = X[column].cat.set_categories(categories)
        return X


class XGBoostModel(TreeModel):
    def new_regressor(self):
        return XGBRegressor(
            **self.params, enable_categorical=True, device="cpu", random_state=SEED
        )


class CatBoostModel(TreeModel):
    CATEGORICAL_COLUMNS = ["community", "house_type", "house_style", "is_apartment"]

    def new_regressor(self):
        return CatBoostRegressor(
            **self.params,
            loss_function="RMSE",
            eval_metric="MAE",
            bootstrap_type="Bernoulli",  # lets subsample work on both CPU and GPU
            task_type="CPU",
            cat_features=self.CATEGORICAL_COLUMNS,
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        )

    def prepare(self, X):
        for column in self.CATEGORICAL_COLUMNS:
            X[column] = X[column].astype(str).mask(X[column].isna(), "missing")
        return X


class LightGBMModel(TreeModel):
    def new_regressor(self):
        return LGBMRegressor(**self.params, random_seed=SEED, verbosity=-1)


class LightGBMSplitApartmentModel(TreeModel):
    def fit(self, train):
        X, y = self.training_features(train)
        self.regressors = {}
        for flag, params in self.params.items():
            rows = X["is_apartment"] == flag
            regressor = LGBMRegressor(
                **params, subsample_freq=1, random_state=SEED, verbosity=-1
            )
            self.regressors[flag] = regressor.fit(
                self.without_apartment_columns(X[rows]), y[rows]
            )
        return self

    def predict(self, homes):
        X = self.features_to_value(homes)
        log_price = pd.Series(np.nan, index=X.index)
        for flag, regressor in self.regressors.items():
            rows = X["is_apartment"] == flag
            if rows.any():
                log_price[rows] = regressor.predict(
                    self.without_apartment_columns(X[rows])
                )
        return np.exp(log_price.to_numpy())

    @staticmethod
    def without_apartment_columns(X):
        return X.drop(
            columns=[
                column for column in X.columns if column.startswith("is_apartment")
            ]
        )


class MetaModel:
    def __init__(self, track):
        self.track = track

    def new_base_models(self):
        make_features = make_features_for(self.track)
        params = TRACK_PARAMS[self.track]
        return {
            "xgboost": XGBoostModel(make_features, params["xgboost"]),
            "catboost": CatBoostModel(make_features, params["catboost"]),
            "lightgbm": LightGBMModel(make_features, params["lightgbm"]),
            "lightgbm_split_apartment": LightGBMSplitApartmentModel(
                make_features, params["lightgbm_split_apartment"]
            ),
        }

    def fit(self, train):
        start = time.perf_counter()
        last_month = train["date_sold"].max().to_period("M")
        months = pd.period_range(end=last_month, periods=META_MONTHS, freq="M")
        log_predictions, log_prices = [], []
        for month in months:
            earlier = train[train["date_sold"] < month.start_time]
            homes = train[train["date_sold"].dt.to_period("M") == month]
            log_predictions.append(
                self.base_log_predictions(
                    earlier, data.hide_answers(homes, self.track), f"out-of-fold {month}"
                )
            )
            log_prices.append(np.log(homes["sold_price"]))

        self.ridge = Ridge(alpha=TRACK_PARAMS[self.track]["meta_alpha"]).fit(
            pd.concat(log_predictions), pd.concat(log_prices)
        )
        self.train = train  # the final base models are trained when predict needs them
        log(f"{self.track} meta model trained in {time.perf_counter() - start:.0f}s")
        return self

    def predict(self, homes):
        return np.exp(
            self.ridge.predict(self.base_log_predictions(self.train, homes, "final"))
        )

    def __getstate__(self):
        return {name: value for name, value in self.__dict__.items() if name != "train"}

    def attach_train(self, train):
        """Give a loaded model back the training table it was fitted on."""
        self.train = train
        for model in getattr(self, "fitted_base_models", {}).values():
            model.train = train
        # The table may hash differently on another machine: keep the loaded base models.
        if hasattr(self, "fitted_base_models"):
            self.fitted_train_key = _table_fingerprint(train)
        return self

    def fit_final_base_models(self):
        """Train now the base models that predict would otherwise train when needed."""
        self.base_models_for(self.train, "final")
        return self

    def base_models_for(self, train, label):
        # Keep the last trained base models: predict may be called again on other homes.
        train_key = _table_fingerprint(train)
        if getattr(self, "fitted_train_key", None) != train_key:
            self.fitted_base_models = self.fit_base_models(train, label)
            self.fitted_train_key = train_key
        return self.fitted_base_models

    def base_log_predictions(self, train, homes, label):
        """Log price of `homes` predicted by each base model trained on `train`."""
        key = _fingerprint(
            CODE_FINGERPRINT, self.track, _table_fingerprint(train), _table_fingerprint(homes)
        )
        path = None if CACHE_DIR is None else Path(CACHE_DIR) / f"{key}.parquet"
        if key not in _BASE_LOG_PREDICTIONS and path is not None and path.exists():
            _BASE_LOG_PREDICTIONS[key] = pd.read_parquet(path)
        if key not in _BASE_LOG_PREDICTIONS:
            predictions = self.predict_base_models(self.base_models_for(train, label), homes)
            _BASE_LOG_PREDICTIONS[key] = predictions.reset_index(drop=True)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                _BASE_LOG_PREDICTIONS[key].to_parquet(path)
        return _BASE_LOG_PREDICTIONS[key].set_axis(homes.index)

    def fit_base_models(self, train, label):
        base_models = {}
        for name, model in self.new_base_models().items():
            start = time.perf_counter()
            base_models[name] = model.fit(train)
            total_seconds = time.perf_counter() - start
            log(
                f"  {self.track} {label}: {name} on {len(train):,} sales: "
                f"features {model.feature_seconds:.0f}s, "
                f"training {total_seconds - model.feature_seconds:.0f}s"
            )
        return base_models

    @staticmethod
    def predict_base_models(base_models, homes):
        return pd.DataFrame(
            {name: np.log(model.predict(homes)) for name, model in base_models.items()},
            index=homes.index,
        )
