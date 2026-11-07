import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

APARTMENT_TYPES = ["Co-Op Apt", "Condo Apt", "Condo Townhouse"]
SIZE_COLUMNS = ["square_feet", "bedrooms", "washrooms", "parking"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def is_apartment(homes):
    """1 for condo apartments, co-op apartments and condo townhouses, 0 for other homes."""
    return homes["house_type_name"].isin(APARTMENT_TYPES).astype(int)


def add_size_interactions(X):
    """Size columns split by is_apartment (so apartments and houses get their own slope),
    then every pair of size columns multiplied together."""
    for column in SIZE_COLUMNS:
        for flag in [0, 1]:
            X[f"is_apartment_x_{column}_{flag}"] = (X["is_apartment"] == flag).astype(
                int
            ) * X[column]
    for i, first in enumerate(SIZE_COLUMNS):
        for second in SIZE_COLUMNS[i + 1 :]:
            X[f"{first}_x_{second}"] = X[first] * X[second]
    return X


def price_lags(homes, history=None):
    """Median log sold price of the sales listed exactly a year (364 days) and half a year
    (182 days) before each home was listed. Sales come from `history` when given,
    otherwise from `homes`."""
    source = homes if history is None else history
    daily_median = np.log(source["sold_price"]).groupby(source["date_listed"]).median()
    X = pd.DataFrame(index=homes.index)
    X["lag_annual"] = (homes["date_listed"] - pd.Timedelta(days=364)).map(daily_median)
    X["lag_semi_annual"] = (homes["date_listed"] - pd.Timedelta(days=182)).map(
        daily_median
    )
    return X


# ---- earlier sales in the same building ----

# Condo apartments and townhouses in one building share one exact location, so the
# location and property type identify a building. A repeat sale is the same home sold
# again: same building or lot, same type, rooms and area.
BUILDING_KEY = ["latitude", "longitude", "house_type_name"]
REPEAT_SALE_KEY = BUILDING_KEY + ["bedrooms", "washrooms", "house_area"]


def _key(df, columns):
    """One string per row joining `columns`, rounding coordinates so floats match."""
    parts = [
        df[c].round(6).astype(str)
        if c in ("latitude", "longitude")
        else df[c].astype(str)
        for c in columns
    ]
    return pd.Series(["|".join(row) for row in zip(*parts)], index=df.index)


def market_index(source, months):
    """Median log price per square foot of each property type, by month, as known at the
    start of each month in `months`: every value uses only sales of earlier months.
    Smoothed over three months, and carried forward through months with no sales."""
    sales = pd.DataFrame(
        {
            "house_type_name": source["house_type_name"].to_numpy(),
            "month": source["date_sold"].dt.to_period("M").to_numpy(),
            "lppsf": np.log(
                source["sold_price"].to_numpy()
                / square_feet(source["house_area"]).to_numpy()
            ),
        }
    )
    monthly = sales.groupby(["month", "house_type_name"])["lppsf"].median().unstack()
    every_month = pd.period_range(
        min(monthly.index.min(), months.min()),
        max(monthly.index.max(), months.max()),
        freq="M",
    )
    smoothed = monthly.reindex(every_month).rolling(3, min_periods=1).median().ffill()
    return smoothed.shift(1)  # the value for a month comes from the months before it


def _index_at(index, house_types, months):
    """Look up `index` for each (house type, month) pair."""
    keys = pd.MultiIndex.from_arrays(
        [pd.PeriodIndex(months, freq="M"), np.asarray(house_types)]
    )
    return index.stack().reindex(keys).to_numpy()


def building_features(homes, history=None):
    """Earlier sales in the same building, and an earlier sale of the same home.

    Prices of earlier sales are moved to the home's listing month with the market
    index of its property type. A sale counts only when it sold before the home was
    listed, so a training row never sees its own sale or a later one. Sales come
    from `history` when given, otherwise from `homes`.
    """
    source = homes if history is None else history
    listed_months = homes["date_listed"].dt.to_period("M")
    index = market_index(source, listed_months)

    sold_months = source["date_sold"].dt.to_period("M")
    sold_lppsf = np.log(source["sold_price"] / square_feet(source["house_area"]))
    # Log price per square foot with the market level of its sale month taken out.
    detrended = sold_lppsf - _index_at(index, source["house_type_name"], sold_months)
    level_now = _index_at(index, homes["house_type_name"], listed_months)
    log_sqft = np.log(square_feet(homes["house_area"]))

    def earlier_sales(columns):
        sales = (
            pd.DataFrame(
                {
                    "key": _key(source, columns),
                    "date_sold": source["date_sold"],
                    "detrended": detrended,
                }
            )
            .dropna(subset=["detrended"])
            .sort_values("date_sold")
        )
        grouped = sales.groupby("key")["detrended"]
        sales["mean"] = grouped.cumsum() / (grouped.cumcount() + 1)
        sales["count"] = grouped.cumcount() + 1
        queries = pd.DataFrame(
            {
                "key": _key(homes, columns),
                "date_listed": homes["date_listed"],
                "row": np.arange(len(homes)),
            }
        ).sort_values("date_listed")
        matched = pd.merge_asof(
            queries,
            sales,
            left_on="date_listed",
            right_on="date_sold",
            by="key",
            allow_exact_matches=False,  # sold strictly before the home was listed
        )
        return matched.sort_values("row").set_index(homes.index)

    X = pd.DataFrame(index=homes.index)
    building = earlier_sales(BUILDING_KEY)
    X["building_sales"] = building["count"].fillna(0).to_numpy()
    X["building_log_ppsf"] = (building["mean"] + level_now).to_numpy()
    X["building_log_price"] = X["building_log_ppsf"] + log_sqft
    X["building_last_log_ppsf"] = (building["detrended"] + level_now).to_numpy()
    X["building_days_since_sale"] = (
        building["date_listed"] - building["date_sold"]
    ).dt.days.to_numpy()

    repeat = earlier_sales(REPEAT_SALE_KEY)
    X["repeat_log_price"] = (repeat["detrended"] + level_now).to_numpy() + log_sqft
    X["repeat_days_since_sale"] = (
        repeat["date_listed"] - repeat["date_sold"]
    ).dt.days.to_numpy()
    return X


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


def _remarks_text(homes):
    return homes["remarks"].fillna("").astype(str).str.lower()


def remarks_keyword_features(homes):
    """A 0/1 column for each keyword group in REMARKS_KEYWORD_GROUPS, plus a condition
    score: good-condition groups found minus needs-work groups found."""
    remarks = _remarks_text(homes)
    X = pd.DataFrame(index=homes.index)
    for name, pattern in REMARKS_KEYWORD_GROUPS.items():
        X[f"remarks_{name}"] = remarks.str.contains(pattern, regex=True).astype(int)
    X["remarks_condition_score"] = X[
        [f"remarks_{g}" for g in CONDITION_POSITIVE_GROUPS]
    ].sum(axis=1) - X[[f"remarks_{g}" for g in CONDITION_NEGATIVE_GROUPS]].sum(axis=1)
    return X


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
TEXT_SCORE_MIN_TRAIN = 1_000
TEXT_MODEL_CACHE_SIZE = 16
_TEXT_MODELS = OrderedDict()


def _make_remarks_cleaner(communities):
    """Numbers and number words become NUM, community names become COMMUNITY.
    The communities come from the training sales only."""
    names = sorted({c for c in communities if c}, key=len, reverse=True)
    community_re = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b")
    number_word_re = re.compile(r"\b(?:" + "|".join(TEXT_NUMBER_WORDS) + r")\b")

    def clean(text):
        text = re.sub(r"\d[\d,\.]*", " NUM ", text.lower())
        text = community_re.sub(" COMMUNITY ", text)
        return number_word_re.sub(" NUM ", text)

    return clean


def _training_key(train):
    """Identifies a set of training sales, so its text model is fitted once."""
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

    With `history`, the model is fitted on those sales and predicts `homes`. Without
    `history`, `homes` are the training sales: each gets an out-of-fold prediction, so
    the tree models learn how far to trust the score on remarks the Ridge has not seen.
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


# ---- the feature tables ----


TRAINING_FEATURE_CACHE_SIZE = 2
_TRAINING_FEATURES = OrderedDict()


def off_market_features(homes, history=None):
    """Features for homes with no list price.

    Training features (no `history`) are kept for the last few training sets, since
    every base model of both tracks builds the same table from the same sales."""
    if history is not None:
        return _off_market_features(homes, history)
    key = _training_key(homes)
    if key in _TRAINING_FEATURES:
        _TRAINING_FEATURES.move_to_end(key)
    else:
        _TRAINING_FEATURES[key] = _off_market_features(homes)
        if len(_TRAINING_FEATURES) > TRAINING_FEATURE_CACHE_SIZE:
            _TRAINING_FEATURES.popitem(last=False)
    return _TRAINING_FEATURES[key].copy()


def _off_market_features(homes, history=None):
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
    X["month"] = homes["date_listed"].dt.month
    X["year"] = homes["date_listed"].dt.year
    X["day"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days
    X["is_apartment"] = pd.Categorical(is_apartment(homes), categories=[0, 1])
    X = add_size_interactions(X)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X = X.join(price_lags(homes, history))
    X = X.join(building_features(homes, history))
    X = X.join(remarks_keyword_features(homes))
    X["remarks_text_score"] = remarks_text_score(homes, history)
    return X


def list_aware_features(homes, history=None):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes, history)
    X["list_price"] = homes["list_price"]
    return X
