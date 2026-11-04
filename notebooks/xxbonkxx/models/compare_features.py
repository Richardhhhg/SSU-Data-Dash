"""Walk-forward comparison of the XGBoost model with and without the building and
smoothed-lag features in xxbonkxx_features.py. From this folder:

    python compare_features.py                       off-market, every variant
    python compare_features.py list_aware            list-aware instead
    python compare_features.py off_market base both  only some variants

Uses the tuned XGBoost settings from baseline_training.ipynb and every feature (no
feature selection), so the variants differ only in the features switched on. Each
validation month (December 2025 to May 2026) is predicted by a model trained on the
sales before it, like `python run.py validate`.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")

import data
import xxbonkxx_features
from training_utils import walk_forward_validate

SEED = 987
# Best Optuna trial in baseline_training.ipynb, tuned on July to November 2025.
XGB_PARAMS = {
    "learning_rate": 0.06,
    "max_depth": 8,
    "n_estimators": 2000,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "max_bin": 730,
    "min_child_weight": 6,
    "gamma": 0.04,
    "reg_lambda": 0.7489731410235859,
    "reg_alpha": 0.8186696228422924,
    "max_delta_step": 5,
}
# Variant name: (USE_BUILDING_FEATURES, SMOOTH_LAGS).
VARIANTS = {
    "base": (False, False),
    "smooth_lags": (False, True),
    "building": (True, False),
    "both": (True, True),
}


def fit_model(X_train, y_train):
    model = XGBRegressor(**XGB_PARAMS, enable_categorical=True, random_state=SEED)
    return model.fit(X_train, y_train, verbose=False)


def main():
    track = sys.argv[1] if len(sys.argv) > 1 else "off_market"
    names = sys.argv[2:] or list(VARIANTS)
    make_features = (xxbonkxx_features.off_market_features if track == "off_market"
                     else xxbonkxx_features.list_aware_features)
    listings = data.load("listings")

    scores = {}
    for name in names:
        xxbonkxx_features.USE_BUILDING_FEATURES, xxbonkxx_features.SMOOTH_LAGS = VARIANTS[name]
        print(f"\n== {name} ==", flush=True)
        result = walk_forward_validate(listings, make_features, fit_model, track)
        error = 100 * np.abs(result.predicted / result.actual - 1)
        scores[name] = (error.median(), error.mean())

    print(f"\n{track}, walk-forward December 2025 to May 2026")
    print(f"{'variant':<12} {'MdAPE':>7} {'MAPE':>7}")
    for name, (mdape, mape) in scores.items():
        print(f"{name:<12} {mdape:6.2f}% {mape:6.2f}%")


if __name__ == "__main__":
    main()
