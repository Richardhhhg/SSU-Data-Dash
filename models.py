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

import time

import numpy as np
import pandas as pd
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

# colsample_bytree and feature_fraction are the same setting; LightGBM uses feature_fraction.
LIGHTGBM_PARAMS = {
    "learning_rate": 0.08,
    "num_leaves": 160,
    "max_depth": 12,
    "n_estimators": 1000,
    "colsample_bytree": 1.0,
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

META_MONTHS = 6  # months of out-of-fold predictions the meta model learns from
META_ALPHA = 1.931  # ridge penalty chosen by RidgeCV in meta_training.ipynb
VERBOSE = True  # print how long each model takes to train


def log(message):
    if VERBOSE:
        print(message, flush=True)


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    # return ListPriceBaseline() if track == "list_aware" else NearestSalesBaseline(k=10)
    return MetaModel(track)


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
    """What the gradient-boosted tree models below share.

    They learn log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. The training sales are kept, because
    the features of the homes to value (price lags, earlier sales in the same building,
    the remarks text score) are worked out from them. Each model only says which
    regressor it trains, in new_regressor().
    """

    def __init__(self, make_features):
        self.make_features = make_features

    def fit(self, train):
        X, y = self.training_features(train)
        self.regressor = self.new_regressor().fit(X, y)
        return self

    def predict(self, homes):
        return np.exp(self.regressor.predict(self.features_to_value(homes)))

    def training_features(self, train):
        """Features and log sold prices of the training sales. Remembers the sales and
        the categories seen in them, for features_to_value(), and how long building
        the features took, for the timing log."""
        self.train = train
        start = time.perf_counter()
        X = self.make_features(train)
        self.feature_seconds = time.perf_counter() - start
        self.categories = {
            column: X[column].cat.categories for column in X.select_dtypes("category")
        }
        return self.prepare(X), np.log(train["sold_price"])

    def features_to_value(self, homes):
        """Features of homes to value, worked out from the training sales."""
        return self.prepare(self.make_features(homes, history=self.train))

    def prepare(self, X):
        """Give the categorical columns the categories of the training sales, so a code
        means the same category in training and prediction. Unseen values become missing."""
        for column, categories in self.categories.items():
            X[column] = X[column].cat.set_categories(categories)
        return X


class XGBoostModel(TreeModel):
    """XGBoost on the features in features.py, reading the categorical columns natively."""

    def new_regressor(self):
        return XGBRegressor(
            **XGBOOST_PARAMS, enable_categorical=True, device="cpu", random_state=SEED
        )


class CatBoostModel(TreeModel):
    """CatBoost on the features in features.py.

    CatBoost wants its categorical columns as text, and rejects missing values in them,
    so missing values become "missing".
    """

    CATEGORICAL_COLUMNS = ["community", "house_type", "house_style", "is_apartment"]

    def new_regressor(self):
        return CatBoostRegressor(
            **CATBOOST_PARAMS,
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
    """LightGBM on the features in features.py, reading the categorical columns natively."""

    def new_regressor(self):
        return LGBMRegressor(**LIGHTGBM_PARAMS, random_seed=SEED, verbosity=-1)


class LightGBMSplitApartmentModel(TreeModel):
    """Two LightGBM models: one trained only on apartments (condo apartments, co-op
    apartments and condo townhouses), one only on houses. Each home is priced by the
    model of its own kind.

    Both use the features in features.py, built on all the training sales, so price
    lags and the remarks text score still learn from both kinds. The is_apartment
    columns are dropped, as they are the same for every home a model sees.
    """

    PARAMS = {1: LIGHTGBM_APARTMENT_PARAMS, 0: LIGHTGBM_HOUSE_PARAMS}  # by is_apartment

    def fit(self, train):
        X, y = self.training_features(train)
        self.regressors = {}
        for flag, params in self.PARAMS.items():
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
    """A ridge regression stacking the four tree models' predictions of log(price).

    The ridge must learn how far to trust each tree model from predictions on sales
    the tree models never saw. So each of the META_MONTHS months before the end of the
    training sales is predicted the way `python run.py validate` predicts a month: by
    tree models trained only on the sales before that month, with the answers hidden.
    The ridge learns from those predictions, then the tree models are trained again on
    every training sale to price the homes to value.
    """

    def __init__(self, track):
        self.track = track

    def new_base_models(self):
        make_features = make_features_for(self.track)
        return {
            "xgboost": XGBoostModel(make_features),
            "catboost": CatBoostModel(make_features),
            "lightgbm": LightGBMModel(make_features),
            "lightgbm_split_apartment": LightGBMSplitApartmentModel(make_features),
        }

    def fit(self, train):
        start = time.perf_counter()
        last_month = train["date_sold"].max().to_period("M")
        months = pd.period_range(end=last_month, periods=META_MONTHS, freq="M")
        log_predictions, log_prices = [], []
        for month in months:
            earlier = train[train["date_sold"] < month.start_time]
            homes = train[train["date_sold"].dt.to_period("M") == month]
            base_models = self.fit_base_models(earlier, f"out-of-fold {month}")
            log_predictions.append(
                self.base_log_predictions(
                    base_models, data.hide_answers(homes, self.track)
                )
            )
            log_prices.append(np.log(homes["sold_price"]))

        self.ridge = Ridge(alpha=META_ALPHA).fit(
            pd.concat(log_predictions), pd.concat(log_prices)
        )
        self.base_models = self.fit_base_models(train, "final")
        log(f"{self.track} meta model trained in {time.perf_counter() - start:.0f}s")
        return self

    def predict(self, homes):
        return np.exp(
            self.ridge.predict(self.base_log_predictions(self.base_models, homes))
        )

    def fit_base_models(self, train, label):
        """Train every tree model on `train`. `label` names this step in the timing log."""
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
    def base_log_predictions(base_models, homes):
        """One column of log price predictions per tree model, one row per home."""
        return pd.DataFrame(
            {name: np.log(model.predict(homes)) for name, model in base_models.items()},
            index=homes.index,
        )
