lightgbm_best_params = {
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

lightgbm_split_house_params = {
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

lightgbm_split_apartment_params = {
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

xgboost_params = {
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

catboost_params = {
    "learning_rate": 0.09999999999999999,
    "depth": 9,
    "iterations": 1400,
    "l2_leaf_reg": 0.040807415431426686,
    "random_strength": 2.5265354019008015,
    "subsample": 1.0,
    "border_count": 254,
    "one_hot_max_size": 7,
}
