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


def oversample_cat(df, feature, amount, seed=None):
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


def oversample_num(df, feature, min_value, max_value, amount, seed=None):
    """Add `amount` copies of rows whose `feature` is between `min_value` and `max_value`.

    Bounds are inclusive. `min_value` None keeps every row at or below `max_value`, and
    `max_value` None keeps every row at or above `min_value`. Copies are drawn with
    replacement. The original rows are kept, and each copy stays next to its original
    so time-based splits still work. Only oversample training data.
    """
    values = df[feature]
    in_range = pd.Series(True, index=df.index)
    if min_value is not None:
        in_range &= values >= min_value
    if max_value is not None:
        in_range &= values <= max_value
    group = df.loc[in_range]
    if group.empty:
        raise ValueError(f"No rows of {feature} fall in the requested range")
    extra = group.sample(amount, replace=True, random_state=seed)
    return pd.concat([df, extra]).sort_index(kind="stable").reset_index(drop=True)


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


def area_was_range(df: pd.DataFrame) -> pd.Series:
    """Return a boolean series indicating whether the house_area was a range."""
    return df["house_area"].astype(str).str.contains("-", regex=False)


def make_price_per_sqft(df: pd.DataFrame) -> pd.Series:
    """Return the price per square foot for each home."""
    return df["sold_price"] / square_feet(df["house_area"])


def make_lot_missing(df: pd.DataFrame) -> pd.Series:
    """Return a boolean series indicating whether the lot_front or lot_depth is missing."""
    return df["lot_front"].isna() | df["lot_depth"].isna()


def add_lags(
    df: pd.DataFrame, target_name: str, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Median of `target_name` by list date, lagged 182 and 364 days.

    Medians come from `history` when given, otherwise from `df`. `df` still needs
    `date_listed`. Pass the fold's training rows as history when featurizing rows
    the model has not trained on.
    """
    df = df.copy()
    source = df if history is None else history
    target_df_daily = source.groupby("date_listed")[target_name].median()
    listed = df["date_listed"]

    df["lag_annual"] = (listed - pd.Timedelta(days=364)).map(target_df_daily)
    df["lag_semi_annual"] = (listed - pd.Timedelta(days=182)).map(target_df_daily)
    return df


def add_month_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_total_rooms(df: pd.DataFrame) -> pd.Series:
    return df["bedrooms"] + df["washrooms"] + df["parking_total"]


def make_room_density(df: pd.DataFrame) -> pd.Series:
    """Return the number of rooms per square foot."""
    return make_total_rooms(df) / (square_feet(df["house_area"]) + 1)


def make_bedroom_density(df: pd.DataFrame) -> pd.Series:
    """Return the number of bedrooms per square foot."""
    return df["bedrooms"] / (square_feet(df["house_area"]) + 1)


def make_bathroom_density(df: pd.DataFrame) -> pd.Series:
    """Return the number of bathrooms per square foot."""
    return df["washrooms"] / (square_feet(df["house_area"]) + 1)


def make_bedroom_to_bathroom_ratio(df: pd.DataFrame) -> pd.Series:
    """Return the ratio of bedrooms to bathrooms."""
    return df["bedrooms"] / (df["washrooms"] + 1)


def make_median_price_per_sqft_by_community(df: pd.DataFrame) -> pd.Series:
    """Return the median price per square foot for each community."""
    price_per_sqft = make_price_per_sqft(df)
    return df.groupby("community")["community"].transform(
        lambda x: price_per_sqft[x.index].median()
    )


def log_sold_price_quantile(sold_price, reference_sold_price):
    """Fraction of the reference sales whose log sold price is at or below each sale.

    The result is in [0, 1]: 0 when a sale is cheaper than every reference sale, 1 when
    it is at least as expensive as every reference sale. Build the reference from
    training sales only, and score validation rows against that same reference.
    """
    log_reference = np.sort(np.log(np.asarray(reference_sold_price, dtype=float)))
    log_price = np.log(np.asarray(sold_price, dtype=float))
    return np.searchsorted(log_reference, log_price, side="right") / len(log_reference)


def log_price_from_quantile(quantiles, reference_sold_price):
    """Log sold price at each quantile of the reference sales.

    Inverse of log_sold_price_quantile, interpolating between adjacent reference
    prices. Quantiles outside [0, 1] are clipped, so this stays inside the cheapest
    and most expensive reference sale.
    """
    log_reference = np.sort(np.log(np.asarray(reference_sold_price, dtype=float)))
    n = len(log_reference)
    quantiles = np.clip(np.asarray(quantiles, dtype=float), 0.0, 1.0)
    position = np.clip(quantiles * n - 1.0, 0.0, n - 1.0)
    lower = np.floor(position).astype(int)
    upper = np.ceil(position).astype(int)
    weight = position - lower
    return log_reference[lower] * (1.0 - weight) + log_reference[upper] * weight


def _lag_history(homes, history):
    """Rows whose sold prices may be used for lags. Defaults to `homes`."""
    source = homes if history is None else history
    return pd.DataFrame(
        {
            "date_listed": source["date_listed"].to_numpy(),
            "log_sold_price": np.log(source["sold_price"].to_numpy()),
        }
    )


def off_market_features(homes, history=None):
    """Features for homes with no list price.

    `history` is the sales whose prices may be used for target-based features
    (the lags). It defaults to `homes`. During cross validation, pass the fold's
    training rows when building features for that fold's validation rows.
    """
    X = pd.DataFrame(index=homes.index)
    # X["house_area_was_range"] = area_was_range(homes).astype(int)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    # X["lot_missing"] = make_lot_missing(homes)
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # X["municipality"] = homes["municipality"].astype("category")
    # Months since January 2021, so the model can follow prices over time.
    X["community"] = homes["community"].astype("category")
    X["house_type"] = homes["house_type_name"].astype("category")
    X["house_style"] = homes["house_style"].astype("category")
    X["is_apartment"] = make_is_apartment(homes).astype("category")
    X["total_rooms"] = make_total_rooms(homes)
    # X["room_density"] = make_room_density(homes)
    X["bedroom_density"] = make_bedroom_density(homes)
    X["bathroom_density"] = make_bathroom_density(homes)
    X["bedroom_to_bathroom_ratio"] = make_bedroom_to_bathroom_ratio(homes)
    X["month"] = make_months(homes)
    X["year"] = make_years(homes)
    X["day"] = make_days(homes)

    X = add_month_cyclical(X)

    X["date_listed"] = homes["date_listed"]
    X = add_lags(X, "log_sold_price", history=_lag_history(homes, history))
    X.drop(columns=["date_listed"], inplace=True)
    return X


def list_aware_features(homes, history=None):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes, history=history)
    X["list_price"] = homes["list_price"]
    return X
