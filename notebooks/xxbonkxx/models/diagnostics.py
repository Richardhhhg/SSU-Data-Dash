import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def percent_error(actual, predicted):
    return 100 * (np.asarray(predicted) / np.asarray(actual) - 1)


def plot_residuals_vs_actual(actual, predicted):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(actual, percent_error(actual, predicted), s=3, alpha=0.3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set(
        xlabel="Actual sale price ($)",
        ylabel="Residual (% of actual)",
        title="Residuals vs actual sale price",
    )
    plt.show()


def plot_predicted_vs_actual(actual, predicted):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=3, alpha=0.3)
    limits = [
        min(np.min(actual), np.min(predicted)),
        max(np.max(actual), np.max(predicted)),
    ]
    ax.plot(limits, limits, color="black", linewidth=1, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlabel="Actual sale price ($)",
        ylabel="Predicted sale price ($)",
        title="Predicted vs actual sale price",
    )
    ax.legend()
    plt.show()


def plot_residuals_by_feature(
    X, actual, predicted, min_num_cat=10, max_categories=20, n_cols=3
):
    residual = percent_error(actual, predicted)
    n_rows = int(np.ceil(len(X.columns) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )

    for ax, column in zip(axes.flat, X.columns):
        values = X[column].to_numpy()
        is_numeric = pd.api.types.is_numeric_dtype(X[column])
        if is_numeric and X[column].nunique() > min_num_cat:
            ax.scatter(values, residual, s=3, alpha=0.3)
        else:
            order = X[column].value_counts().index[:max_categories].tolist()
            if is_numeric:
                order = sorted(order)
            sns.boxplot(x=values, y=residual, order=order, showfliers=False, ax=ax)
            ax.tick_params(axis="x", rotation=90)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(title=column, xlabel="", ylabel="Residual (%)")

    for ax in axes.flat[len(X.columns) :]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()


def worst_predictions(homes, actual, predicted, n=20):
    table = homes.assign(
        actual_price=np.asarray(actual),
        predicted_price=np.round(predicted),
        error_pct=percent_error(actual, predicted).round(1),
    )
    return table.loc[table["error_pct"].abs().nlargest(n).index]
