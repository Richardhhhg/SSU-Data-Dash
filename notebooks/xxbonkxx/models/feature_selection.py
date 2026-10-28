import numpy as np
import pandas as pd


def _print_scores(label, scorer_name, scores):
    mdape, mape = scores
    print(f"{label:<10} {scorer_name:<9} MdAPE {mdape:.2f}%  MAPE {mape:.2f}%")


def _report(all_features, selected, label, select_on, report_on):
    for scorer_name, scorer in [("select", select_on), ("report", report_on)]:
        if scorer is None:
            continue
        _print_scores("Baseline", scorer_name, scorer(all_features))
        _print_scores(label, scorer_name, scorer(selected))


def feature_importances(scorer):
    """Importances of fit_model trained on the earliest selection fold's training
    sales, so the ranking never sees a month that either scorer predicts."""
    fold = scorer.folds[0]
    model = scorer.fit_model(fold.X_train, fold.y_train)
    return pd.Series(
        model.feature_importances_, index=fold.X_train.columns
    ).sort_values(ascending=False)


def select_top_k_features(select_on, k, report_on=None):
    """Keep the k most important features and compare them with all features, month by month."""
    importances = feature_importances(select_on)
    all_features = list(importances.index)
    top_k_features = list(importances.index[:k])

    print(f"Selected Features: {top_k_features}")
    _report(all_features, top_k_features, f"Top-{k}", select_on, report_on)
    return top_k_features


def backward_feature_elimination(
    select_on, report_on=None, features=None, max_rounds=None
):
    """Repeatedly drop the feature whose removal most lowers walk-forward MdAPE on
    `select_on`; stop when no removal helps (or after max_rounds removals).

    Each round refits one model per month for every remaining feature, so a round
    costs len(features) * len(select_on.folds) fits.
    """
    current_features = list(
        select_on.folds[0].X_train.columns if features is None else features
    )
    all_features = list(current_features)
    best_score = select_on(current_features)[0]
    print(
        f"Start: {len(current_features)} features, walk-forward MdAPE {best_score:.4f}"
    )

    n_round = 0
    while len(current_features) > 1 and (max_rounds is None or n_round < max_rounds):
        n_round += 1
        print(
            f"Round {n_round}: trying {len(current_features)} removals "
            f"({len(current_features) * len(select_on.folds)} fits)",
            flush=True,
        )
        scores = []
        for feature in current_features:
            temp_features = [f for f in current_features if f != feature]
            temp_score = select_on(temp_features)[0]
            scores.append((feature, temp_score))
            if temp_score < best_score:
                print(
                    f"Feature: {feature} | Score improved from {best_score:.4f} to {temp_score:.4f}",
                    flush=True,
                )

        worst_feature, lowest_feature_score = min(scores, key=lambda x: x[1])
        if lowest_feature_score < best_score:
            print(
                f"Removing feature: {worst_feature} | Score improved from {best_score:.4f} "
                f"to {lowest_feature_score:.4f}",
                flush=True,
            )
            current_features.remove(worst_feature)
            best_score = lowest_feature_score
        else:
            print("No further improvement. Stopping feature elimination.")
            break

    print(f"Final selected features: {current_features}")
    _report(all_features, current_features, "Selected", select_on, report_on)
    return current_features
