# Off Market
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

# List Aware
aware_cat_params = {
    "learning_rate": 0.05,
    "depth": 10,
    "iterations": 2000,
    "l2_leaf_reg": 4.809832193856093,
    "random_strength": 3.0009335320617767,
    "subsample": 0.9,
    "border_count": 254,
    "one_hot_max_size": 9,
}

aware_light_split_house = {
    "learning_rate": 0.03,
    "num_leaves": 212,
    "max_depth": 6,
    "n_estimators": 1800,
    "colsample_bytree": 0.5,
    "min_child_samples": 25,
    "min_split_gain": 0.0,
    "reg_lambda": 0.05107704444922434,
    "reg_alpha": 2.8015668038585706,
}

aware_light_split_apartment = {
    "learning_rate": 0.060000000000000005,
    "num_leaves": 49,
    "max_depth": 13,
    "n_estimators": 1000,
    "colsample_bytree": 0.8,
    "min_child_samples": 82,
    "min_split_gain": 0.0,
    "reg_lambda": 0.6933877086258585,
    "reg_alpha": 0.01197099282824542,
}

aware_light = {
    "learning_rate": 0.04,
    "num_leaves": 235,
    "max_depth": 6,
    "n_estimators": 2000,
    "colsample_bytree": 0.30000000000000004,
    "min_child_samples": 22,
    "min_split_gain": 0.0,
    "reg_lambda": 0.824461364065615,
    "reg_alpha": 0.005510621165738579,
}

aware_xgb = {
    "learning_rate": 0.02,
    "max_depth": 13,
    "n_estimators": 1200,
    "subsample": 0.7000000000000001,
    "colsample_bytree": 0.9,
    "max_bin": 1523,
    "min_child_weight": 5,
    "gamma": 0.01,
    "reg_lambda": 0.09067060845188568,
    "reg_alpha": 2.1221007442196798,
    "max_delta_step": 7,
}

aware_ridge_alpha = 0.3728
