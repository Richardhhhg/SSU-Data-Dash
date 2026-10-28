from dataclasses import dataclass

import numpy as np
import pandas as pd

import data

VALIDATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


def align_categories(X_tr, X_va):
    """Give X_va's categorical columns the categories of X_tr. Unseen values become missing."""
    X_va = X_va.copy()
    for col in X_tr.select_dtypes("category"):
        X_va[col] = X_va[col].cat.set_categories(X_tr[col].cat.categories)
    return X_va


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # one row per validation home, indexed like listings
    X_val: pd.DataFrame  # features of the validation homes, same index
    model: object  # model trained for the last month
    X_train: pd.DataFrame  # features the last model was trained on
    y_train: pd.Series  # log sold prices the last model was trained on

    @property
    def actual(self):
        return self.predictions["sold_price"]

    @property
    def predicted(self):
        return self.predictions["predicted_price"]


@dataclass
class MonthFold:
    month: pd.Period
    track: str
    homes: pd.DataFrame  # the month's sales, answers included, for scoring
    X_train: pd.DataFrame  # features of every earlier sale
    y_train: pd.Series  # log sold prices of every earlier sale
    X_homes: pd.DataFrame  # features of the month's homes, answers hidden


def walk_forward_folds(listings, make_features, track, months=VALIDATION_MONTHS):
    folds = []
    for month in months:
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        X_train = make_features(train)
        X_homes = align_categories(
            X_train, make_features(data.hide_answers(homes, track), history=train)
        )
        folds.append(
            MonthFold(
                month, track, homes, X_train, np.log(train["sold_price"]), X_homes
            )
        )
    return folds


def _predict_month(fold, fit_model, features=None):
    columns = fold.X_train.columns if features is None else list(features)
    model = fit_model(fold.X_train[columns], fold.y_train)
    homes = fold.homes
    predicted = np.round(np.exp(model.predict(fold.X_homes[columns])))
    predictions = pd.DataFrame(
        {
            "ml_num": homes["ml_num"],
            "track": fold.track,
            "month": str(fold.month),
            "house_type_name": homes["house_type_name"],
            "sold_price": homes["sold_price"],
            "predicted_price": predicted,
        },
        index=homes.index,
    )
    return model, predictions


def _percent_errors(predictions):
    return 100 * np.abs(predictions["predicted_price"] / predictions["sold_price"] - 1)


def walk_forward_validate(
    listings,
    make_features,
    fit_model,
    track,
    months=VALIDATION_MONTHS,
    verbose=True,
    folds=None,
):
    if folds is None:
        folds = walk_forward_folds(listings, make_features, track, months)
    predictions, features = [], []
    for fold in folds:
        model, month_predictions = _predict_month(fold, fit_model)
        predictions.append(month_predictions)
        features.append(fold.X_homes)
        if verbose:
            error = _percent_errors(month_predictions)
            print(
                f"{fold.month}: trained on {len(fold.X_train):,} earlier sales, predicted "
                f"{len(fold.homes):,} homes  MdAPE {error.median():.2f}%  MAPE {error.mean():.2f}%",
                flush=True,
            )

    result = WalkForwardResult(
        pd.concat(predictions), pd.concat(features), model, fold.X_train, fold.y_train
    )
    if verbose:
        error = 100 * np.abs(result.predicted / result.actual - 1)
        print(
            f"{folds[0].track}  MdAPE {error.median():.2f}%  MAPE {error.mean():.2f}%"
        )
    return result


class WalkForwardScorer:
    """Walk-forward (MdAPE, MAPE) of fit_model restricted to some feature columns.

    Features come prebuilt in `folds`, so each call only refits one model per month.
    Scores are cached per feature set.
    """

    def __init__(self, folds, fit_model):
        self.folds = folds
        self.fit_model = fit_model
        self._scores = {}

    def __call__(self, features=None):
        key = tuple(self.folds[0].X_train.columns if features is None else features)
        if key not in self._scores:
            predictions = pd.concat(
                [_predict_month(fold, self.fit_model, key)[1] for fold in self.folds]
            )
            error = _percent_errors(predictions)
            self._scores[key] = (float(error.median()), float(error.mean()))
        return self._scores[key]
