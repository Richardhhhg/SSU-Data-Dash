import re
from collections import OrderedDict
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

import features

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


REMARKS_KEYWORD_GROUPS = {
    # Condition: good
    "renovated": r"renovated|redone|rebuilt|refinished",
    "updated": r"updated electrical|updated kitchen|updated bathroom|has been updated|updated flooring|new flooring",
    "turnkey": r"turnkey|nothing to do but move in|move-in ready",
    "kitchen": r"kitchen",
    "premium_kitchen": r"custom kitchen|quartz|large island|stainless steel|new (?:\w+ )?appliances|modern vanity",
    "wood_floor": r"new vinyl plank|hardwood|laminate flooring",
    "clean": r"clean|well cared for|well maintained",
    # Condition: needs work
    "needs_work": r"as-is|needs paint|needs new flooring|needs a full|full cosmetic overhaul|needs updating|\bto update|can update",
    "dated": r"dated|\bolder\b|might need attention|\bold\b",
    # Interior features
    "split_bedroom": r"split-bedroom",
    "open_concept": r"open-concept",
    "walk_in_closet": r"walk-in closet",
    "ensuite_laundry": r"ensuite laundry",
    "den_office": r"\bden\b|\boffice\b",
    "loft": r"\bloft\b",
    "basement_good": r"finished basement|renovated basement",
    "basement_unfinished": r"unfinished basement",
    "carpet": r"broadloom|carpet",
    "hardwood": r"parquet|hardwood",
    "rooftop_terrace": r"rooftop terrace",
    "deck": r"balcony|\bdeck\b",
    "patio": r"patio",
    "porch": r"front porch",
    "backyard": r"backyard",
    "luxury": r"luxury|custom",
    "big_window": r"floor-to-ceiling windows|large windows",
    # Parking and amenities
    "guest_suites": r"guest suite",
    "locker": r"locker",
    "gym": r"\bgym\b",
    # Location
    "downtown_commute": r"downtown commute",
    "highway": r"highway",
    "transit": r"transit",
    "schools": r"schools?\b",
    "parks_trails": r"parks? and trails|\btrails?\b",
    "places_of_worship": r"places of worship",
    "community_centre": r"community centre",
    "shops": r"caf[eé]s?|restaurants|\bshops\b|grocery",
    # Target buyer
    "investor": r"investor|rent or hold",
    "first_time_buyer": r"first-time buyer",
    "family": r"growing family|\bfamil(?:y|ies)\b",
    "couple": r"\bcouples?\b",
    "young_professional": r"young professional",
    # Other
    "studio": r"\bstudio\b",
    "has_street": r"\b(?:way|boulevard|street|avenue|court|drive|road|place|trail|lane|plaza|gardens|crescent)\b",
}

CONDITION_POSITIVE_GROUPS = [
    "turnkey",
    "premium_kitchen",
    "renovated",
    "updated",
    "clean",
    "luxury",
]
CONDITION_NEGATIVE_GROUPS = ["needs_work", "dated"]


def make_remarks_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return 0/1 columns for each keyword group in REMARKS_KEYWORD_GROUPS, plus a net condition score."""
    remarks = df["remarks"].fillna("").astype(str).str.lower()
    features = pd.DataFrame(index=df.index)

    for name, pattern in REMARKS_KEYWORD_GROUPS.items():
        features[f"remarks_{name}"] = remarks.str.contains(pattern, regex=True).astype(
            int
        )

    features["remarks_condition_score"] = features[
        [f"remarks_{g}" for g in CONDITION_POSITIVE_GROUPS]
    ].sum(axis=1) - features[[f"remarks_{g}" for g in CONDITION_NEGATIVE_GROUPS]].sum(
        axis=1
    )

    return features


TEXT_NUMBER_WORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]
TEXT_SCORE_FOLDS = 5
# Below this many training sales the TF-IDF vocabulary (min_df=10) and the folds
# are meaningless, so the score is left missing.
TEXT_SCORE_MIN_TRAIN = 1_000
TEXT_MODEL_CACHE_SIZE = 16
_TEXT_MODELS = OrderedDict()


def _remarks_text(homes):
    return homes["remarks"].fillna("").astype(str).str.lower()


def _make_remarks_cleaner(communities):
    """Preprocessor from remarks_eda: numbers and number words become NUM, community
    names become COMMUNITY. The communities come from the training sales only."""
    names = sorted({c for c in communities if c}, key=len, reverse=True)
    community_re = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")
    number_word_re = re.compile(r"\b(?:" + "|".join(TEXT_NUMBER_WORDS) + r")\b")

    def clean(text):
        text = re.sub(r"\d[\d,\.]*", " NUM ", text.lower())
        text = community_re.sub(" COMMUNITY ", text)
        return number_word_re.sub(" NUM ", text)

    return clean


def _training_key(train):
    """Identifies a set of training sales, so a fold's text model is fitted once."""
    hashes = pd.util.hash_pandas_object(train["ml_num"], index=True).to_numpy()
    return len(train), int(hashes.sum()), int(hashes[::7].sum())


def _fit_text_model(train):
    """TF-IDF + Ridge on the training sales' remarks, predicting log sold price.

    Returns (vectorizer, ridge, out_of_fold) where out_of_fold holds a prediction for
    every training sale from a Ridge that did not see it. The vectorizer uses no
    prices, so it is fitted once on all the training remarks.
    """
    key = _training_key(train)
    if key in _TEXT_MODELS:
        _TEXT_MODELS.move_to_end(key)
        return _TEXT_MODELS[key]

    communities = train["community"].astype(str).str.lower().unique()
    vectorizer = TfidfVectorizer(
        preprocessor=_make_remarks_cleaner(communities),
        ngram_range=(1, 3),
        min_df=10,
        max_df=0.50,
        sublinear_tf=True,
        stop_words=None,  # keep "no", "not", "as", "is" for negation and "as-is"
        max_features=50_000,
    )
    X_text = vectorizer.fit_transform(_remarks_text(train))
    y = np.log(train["sold_price"].to_numpy())

    ridge = Ridge(alpha=1.0).fit(X_text, y)
    folds = KFold(TEXT_SCORE_FOLDS, shuffle=True, random_state=0)
    out_of_fold = pd.Series(
        cross_val_predict(Ridge(alpha=1.0), X_text, y, cv=folds), index=train.index
    )

    _TEXT_MODELS[key] = (vectorizer, ridge, out_of_fold)
    if len(_TEXT_MODELS) > TEXT_MODEL_CACHE_SIZE:
        _TEXT_MODELS.popitem(last=False)
    return _TEXT_MODELS[key]


def remarks_text_score(homes, history=None):
    """Log sold price predicted from the remarks alone by TF-IDF + Ridge.

    With `history`, the model is fitted on those sales and predicts `homes`, so
    validation homes never inform it. Without `history`, `homes` are the training
    sales: each gets an out-of-fold prediction, so the larger model learns how far
    to trust the score on remarks the Ridge has not memorised.
    """
    train = homes if history is None else history
    if len(train) < TEXT_SCORE_MIN_TRAIN:
        return pd.Series(np.nan, index=homes.index)
    vectorizer, ridge, out_of_fold = _fit_text_model(train)
    if history is None:
        return out_of_fold
    return pd.Series(
        ridge.predict(vectorizer.transform(_remarks_text(homes))), index=homes.index
    )


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


def _location_prices(source, log_sold_price):
    return pd.DataFrame(
        {
            "community": source["community"].astype(str).to_numpy(),
            "municipality": source["municipality"].astype(str).to_numpy(),
            "date_listed": source["date_listed"].to_numpy(),
            "log_sold_price": np.asarray(log_sold_price, dtype=float),
        }
    )


def _community_history(homes, history):
    """Rows whose sold prices may be used for community features. Defaults to `homes`."""
    source = homes if history is None else history
    return _location_prices(source, np.log(source["sold_price"].to_numpy()))


def point_in_time_median(known, group_col, groups, dates, window_days=182, min_sales=5):
    """Median log_sold_price of the `known` rows in each row's group, listed in the
    window before that row's date.

    The window is [date - window_days, date), or every earlier date when window_days
    is None. Rows on the date itself are left out. Returns NaN when the window holds
    fewer than `min_sales` sales.
    """
    queries = pd.DataFrame(
        {
            group_col: np.asarray(groups, dtype=str),
            "date_listed": np.asarray(dates, dtype="datetime64[ns]"),
            "log_sold_price": np.nan,
        }
    )
    combined = pd.concat(
        [known[[group_col, "date_listed", "log_sold_price"]], queries],
        ignore_index=True,
    )
    if window_days is None:
        span = combined["date_listed"].max() - combined["date_listed"].min()
        window_days = span.days + 1
    # Groups are contiguous after sorting, so the rolling output keeps this row order.
    order = np.lexsort(
        (combined["date_listed"].to_numpy(), combined[group_col].to_numpy())
    )
    combined = combined.iloc[order]
    medians = (
        combined.groupby(group_col, sort=False)
        .rolling(
            f"{window_days}D", on="date_listed", closed="left", min_periods=min_sales
        )["log_sold_price"]
        .median()
        .to_numpy()
    )
    result = np.empty(len(combined))
    result[order] = medians
    return result[len(known) :]


STREET_CSV_DIR = Path(__file__).resolve().parent.parent / "analysis"
STREET_CSV_FILES = [
    "remarks_with_streets_filtered.csv",
    "off_market_with_streets_filtered.csv",
    "list_aware_with_streets_filtered.csv",
]
STREET_SUFFIX_ABBREVIATIONS = {
    "ave": "avenue",
    "blvd": "boulevard",
    "st": "street",
    "cres": "crescent",
    "cresc": "crescent",
    "rd": "road",
    "dr": "drive",
    "ct": "court",
    "ln": "lane",
    "pl": "place",
    "trails": "trail",
}
STREET_SUFFIXES = r"way|boulevard|street|avenue|court|drive|road|place|trail|lane|plaza|gardens|crescent"
STREET_PATTERN = re.compile(rf"\b([a-z]+) ({STREET_SUFFIXES})\b")
# Words the LLM sometimes returned in front of a suffix that are not street names.
NOT_STREET_NAMES = {"quiet", "lined", "york", "nicest", "the", "a"}


def clean_street_name(raw: str) -> str:
    """First "<name> <suffix>" in an LLM answer, with abbreviated suffixes spelled out."""
    words = re.sub(r"[^a-z\s]", " ", str(raw).lower()).split()
    text = " ".join(STREET_SUFFIX_ABBREVIATIONS.get(w, w) for w in words)
    for name, suffix in STREET_PATTERN.findall(text):
        if name not in NOT_STREET_NAMES:
            return f"{name} {suffix}"
    return "unknown"


@cache
def load_street_names() -> pd.Series:
    """Cleaned LLM street name for every ml_num in the street CSVs."""
    frames = [
        pd.read_csv(STREET_CSV_DIR / name, usecols=["ml_num", "street_llm"])
        for name in STREET_CSV_FILES
    ]
    streets = pd.concat(frames).drop_duplicates("ml_num").set_index("ml_num")
    return streets["street_llm"].map(clean_street_name)


def make_street(homes: pd.DataFrame) -> pd.Series:
    """Street name pulled from the remarks, or "unknown" when none was found."""
    return homes["ml_num"].map(load_street_names()).fillna("unknown")


def community_features(
    homes, history=None, window_days=182, min_sales=5, predict_log_price=None
):
    known = _community_history(homes, history)
    if history is not None and predict_log_price is not None:
        predicted = _location_prices(homes, predict_log_price(homes))
        known = pd.concat([known, predicted], ignore_index=True)
    dates = homes["date_listed"]
    community = homes["community"]

    def community_median(lag_days):
        return point_in_time_median(
            known,
            "community",
            community,
            dates - pd.Timedelta(days=lag_days),
            window_days,
            min_sales,
        )

    X = pd.DataFrame(index=homes.index)
    X["community_median"] = community_median(0)
    X["community_median_lag_6m"] = community_median(182)
    X["community_median_lag_1y"] = community_median(364)
    X["community_change_6m"] = X["community_median"] - X["community_median_lag_6m"]
    X["community_change_1y"] = X["community_median"] - X["community_median_lag_1y"]

    municipality_median = point_in_time_median(
        known, "municipality", homes["municipality"], dates, window_days, min_sales
    )
    X["community_to_market"] = np.exp(X["community_median"] - municipality_median)
    return X


def off_market_features(homes, history=None, window_days=182, predict_log_price=None):
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = np.log(square_feet(homes["house_area"]) + 1)
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
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

    X = add_month_cyclical(X)

    # X["house_area_was_range"] = area_was_range(homes).astype(int)
    # # X["garage"] = homes["garage"].astype(int)
    # # X["total_rooms"] = make_total_rooms(homes)
    # # X["bedroom_density"] = make_bedroom_density(homes)
    # # X["bathroom_density"] = make_bathroom_density(homes)
    # # X["bedroom_to_bathroom_ratio"] = make_bedroom_to_bathroom_ratio(homes)

    X["date_listed"] = homes["date_listed"]
    X = add_lags(X, "log_sold_price", history=_lag_history(homes, history))
    X.drop(columns=["date_listed"], inplace=True)
    X = X.join(
        community_features(
            homes,
            history=history,
            window_days=window_days,
            predict_log_price=predict_log_price,
        )
    )

    keyword_features = make_remarks_keyword_features(homes)
    X = X.join(keyword_features)
    X["remarks_text_score"] = remarks_text_score(homes, history)
    return X


def list_aware_features(homes, history=None, window_days=182, predict_log_price=None):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(
        homes,
        history=history,
        window_days=window_days,
        predict_log_price=predict_log_price,
    )
    X["list_price"] = homes["list_price"]
    return X
