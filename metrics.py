import numpy as np
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool
import pandas as pd

def daily_average_precision(
    y_true,
    y_score,
    assignment_date,
    skip_days_without_positives: bool = True,
    return_per_day: bool = False,
):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_score": np.asarray(y_score),
        "date": np.asarray(assignment_date),
    })

    per_day = {}
    for day, group in df.groupby("date", sort=True):
        n_pos = int(group["y_true"].sum())

        if n_pos == 0:
            if skip_days_without_positives:
                continue
            per_day[day] = 0.0
            continue

        per_day[day] = average_precision_score(group["y_true"], group["y_score"])

    per_day_series = pd.Series(per_day, dtype="float64").sort_index()

    if len(per_day_series) == 0:
        daily_ap = float("nan")
    else:
        daily_ap = float(per_day_series.mean())

    if return_per_day:
        return daily_ap, per_day_series
    return daily_ap

def run_cv_fold(
    train_df,
    valid_df,
    feature_columns,
    categorical_columns,
    target_column,
    model_config,
):
    train_pool = Pool(
        train_df[feature_columns],
        train_df[target_column],
        cat_features=categorical_columns,
    )
    valid_pool = Pool(
        valid_df[feature_columns],
        valid_df[target_column],
        cat_features=categorical_columns,
    )
    
    fold_model = CatBoostClassifier(**model_config)
    fold_model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
    
    valid_scores = fold_model.predict_proba(valid_df[feature_columns])[:, 1]
    fold_daily_ap = daily_average_precision(
        y_true=valid_df[target_column],
        y_score=valid_scores,
        assignment_date=valid_df["assignment_date"],
    )
    
    return fold_daily_ap, fold_model.get_best_iteration()