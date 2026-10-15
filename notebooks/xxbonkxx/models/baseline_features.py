from itertools import combinations

import numpy as np
import pandas as pd

PROPERTY_TYPES = [
    "Condo Apt",
    "Detached",
    "Semi-Detached",
    "Condo Townhouse",
    "Freehold Townhouse",
    "Co-Op Apt",
]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def make_months(df):
    """Return a season for a given date."""
    return df["date_listed"].dt.month


def make_days(df):
    """Return the number of days since January 1, 2021 for a given date."""
    return (df["date_listed"] - pd.Timestamp("2021-01-01")).dt.days


def make_years(df):
    """Return the year for a given date."""
    return df["date_listed"].dt.year


def fillna_lots(df):
    """Fill missing lot_front and lot_depth values with the median for that municipality."""
    df["lot_front"] = df["lot_front"].fillna(0)
    df["lot_depth"] = df["lot_depth"].fillna(0)
    return df


def oversample(df, feature, amount, seed=None):
    """Bring every category of `feature` with fewer than `amount` rows up to `amount` rows
    by adding copies of its rows, drawn with replacement. The original rows are kept.
    Rows stay in their original order (listings is sorted by date_sold), with each copy
    next to its original, so time-based splits still work. Only oversample training data."""
    extra = [
        group.sample(amount - len(group), replace=True, random_state=seed)
        for _, group in df.groupby(feature, observed=True)
        if len(group) < amount
    ]
    return pd.concat([df, *extra]).sort_index(kind="stable").reset_index(drop=True)


def build_single_interaction_feature(df: pd.DataFrame, f1: str, f2: str) -> pd.Series:
    """Build an interaction feature for two features f1 and f2."""
    if isinstance(df[f1].dtype, pd.CategoricalDtype) and isinstance(
        df[f2].dtype, pd.CategoricalDtype
    ):
        # Two categoricals: combine the categories
        return (df[f1].astype(str) + "_" + df[f2].astype(str)).astype("category")
    elif isinstance(df[f1].dtype, pd.CategoricalDtype) or isinstance(
        df[f2].dtype, pd.CategoricalDtype
    ):
        # One categorical one numeric
        # One hot encode the categorical and multiply by the numeric
        cat_col = f1 if isinstance(df[f1].dtype, pd.CategoricalDtype) else f2
        num_col = f2 if cat_col == f1 else f1
        return pd.concat(
            [
                (df[cat_col] == cat).astype(int) * df[num_col]
                for cat in df[cat_col].cat.categories
            ],
            axis=1,
        )
    else:
        # Two numerics: multiply them
        return df[f1] * df[f2]


def build_interaction_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build interaction features for all pairs of features in `features`."""
    df = df.copy()

    for f1, f2 in combinations(features, 2):
        interaction_feature = build_single_interaction_feature(df, f1, f2)
        if isinstance(interaction_feature, pd.DataFrame):
            # One categorical and one numeric: add each column separately
            for col in interaction_feature.columns:
                df[f"{f1}_x_{f2}_{col}"] = interaction_feature[col]
        else:
            df[f"{f1}_x_{f2}"] = interaction_feature

    return df


def fit_scaling_offset(groups: pd.Series, y_log: pd.Series) -> pd.Series:
    """Median log target of each category minus the overall median log target.
    Fit this on training rows only, then use apply_scaling_offset on any rows."""
    overall_median = y_log.median()
    return y_log.groupby(groups.astype(str)).median() - overall_median


def apply_scaling_offset(groups: pd.Series, offset: pd.Series) -> np.ndarray:
    """Offset of each row's category. Categories not seen when fitting get 0."""
    return groups.astype(str).map(offset).fillna(0).to_numpy()


def make_is_apartment(df: pd.DataFrame) -> pd.Series:
    def is_house_apartment(row):
        if row["house_type_name"] in ["Co-op Apt", "Condo Apt", "Condo Townhouse"]:
            return 1
        else:
            return 0

    return df.apply(is_house_apartment, axis=1)


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = np.log(square_feet(homes["house_area"]) + 1)
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X = fillna_lots(X)
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["municipality"] = homes["municipality"].astype("category")
    # Months since January 2021, so the model can follow prices over time.
    X["community"] = homes["community"].astype("category")
    X["house_type"] = homes["house_type_name"].astype("category")
    X["house_style"] = homes["house_style"].astype("category")
    X["month"] = make_months(homes)
    X["year"] = make_years(homes)
    X["day"] = make_days(homes)
    X["is_apartment"] = make_is_apartment(homes).astype("category")
    X = build_interaction_features(
        X, ["is_apartment", "square_feet", "bedrooms", "washrooms", "parking"]
    )
    # interaction_columns = list(X.columns)
    # interaction_columns.remove("community")
    # X = build_interaction_features(
    #     X,
    #     interaction_columns,
    # )
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X
