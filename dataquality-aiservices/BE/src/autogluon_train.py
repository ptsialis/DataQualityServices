# autogluon_train.py
import os
import json
import shutil
import tempfile
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from autogluon.tabular import TabularPredictor


def zip_dir(src_dir: str, zip_path: str) -> None:
    """Zippt ein komplettes Verzeichnis (rekursiv) nach zip_path."""
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, src_dir)
                z.write(full, arcname=rel)


def _safe_train_test_split(df: pd.DataFrame, target: str, test_size: float, seed: int):

    y = df[target]
    stratify = y
    
    try:
        vc = y.value_counts(dropna=False)
        if vc.min() < 2:
            stratify = None
    except Exception:
        stratify = None

    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def infer_problem_type(y: pd.Series) -> str:
    y_nonnull = y.dropna()
    n = len(y_nonnull)
    if n == 0:
        return "classification"

    n_unique = int(y_nonnull.nunique(dropna=True))
    unique_ratio = n_unique / max(1, n)

    is_numeric = pd.api.types.is_numeric_dtype(y_nonnull)

    if is_numeric:
        if n_unique <= 20 and unique_ratio <= 0.05:
            return "classification"
        return "regression"

    return "classification"


def train_autogluon(
    df: pd.DataFrame,
    target_variable: str,
    preset: str = "best_quality",
    hpo_mode: str = "off",
    time_limit_s: int = 300,
    seed: int | None = None,
    num_trials: int = 20,
    artifact_path: Optional[str] = None,
):

    TARGET = target_variable
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not in df columns.")

    problem_type = infer_problem_type(df[TARGET])

    train_full_df, test_df = _safe_train_test_split(df, TARGET, test_size=0.2, seed=seed)
    train_df, val_df = _safe_train_test_split(train_full_df, TARGET, test_size=0.2, seed=seed)

    # TEMP path
    tmp_root = tempfile.mkdtemp(prefix="ag_")
    ag_path = os.path.join(tmp_root, "predictor")

    try:
        # HPO Einstellungen
        hyperparameter_tune_kwargs = None
        if hpo_mode == "random":
            hyperparameter_tune_kwargs = {
                "num_trials": int(num_trials),
                "searcher": "random",
                "scheduler": "local",
            }
        elif hpo_mode == "grid":
            hyperparameter_tune_kwargs = {
                "num_trials": int(num_trials),
                "searcher": "grid",
                "scheduler": "local",
            }
        elif hpo_mode not in ("off", "none", ""):
            raise ValueError('hpo_mode must be "off", "random", or "grid"')


        if problem_type == "regression":
            eval_metric = "mae"
        else:
            eval_metric = "accuracy"

        predictor = TabularPredictor(
            label=TARGET,
            problem_type="regression" if problem_type == "regression" else None,
            eval_metric=eval_metric,
            path=ag_path,
            verbosity=0,
        )

        fit_kwargs = dict(
            train_data=train_df,
            tuning_data=val_df,
            presets=preset,
            time_limit=int(time_limit_s),
            verbosity=0,
            use_bag_holdout=True,
        )

        if hyperparameter_tune_kwargs is not None:
            fit_kwargs["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs

        predictor = predictor.fit(**fit_kwargs)

        y_true = test_df[TARGET]
        X_test = test_df.drop(columns=[TARGET])
        y_pred = predictor.predict(X_test)

        # Best model name
        lb = predictor.leaderboard(silent=True)
        best_model_name = None
        if lb is not None and len(lb) > 0 and "model" in lb.columns:
            best_model_name = str(lb.iloc[0]["model"])

        # Artefakt ZIP
        if artifact_path:
            zip_dir(ag_path, artifact_path)

        if problem_type == "regression":
            y_true_num = pd.to_numeric(y_true, errors="coerce")
            y_pred_num = pd.to_numeric(pd.Series(y_pred), errors="coerce")
            mask = (~y_true_num.isna()) & (~y_pred_num.isna())
            yt = y_true_num[mask].to_numpy()
            yp = y_pred_num[mask].to_numpy()

            mae = float(mean_absolute_error(yt, yp)) if len(yt) else float("nan")
            rmse = float(np.sqrt(mean_squared_error(yt, yp))) if len(yt) else float("nan")
            r2 = float(r2_score(yt, yp)) if len(yt) else float("nan")

            return {
                "selected_model": "AutoML (AutoGluon)",
                "model_key": "automl",
                "problem_type": "regression",
                "preset": preset,
                "hpo_mode": "off" if hpo_mode in ("off", "none", "") else hpo_mode,
                "time_limit_s": int(time_limit_s),
                "seed": int(seed),
                "best_model_name": best_model_name,
                "metrics": {
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "test_r2": r2,
                },
                "split_data": {
                    "train_table": train_df.to_dict(orient="records"),
                    "validation_table": val_df.to_dict(orient="records"),
                    "test_table": test_df.to_dict(orient="records"),
                },
            }

        else:
            acc = float(accuracy_score(y_true, y_pred))
            f1m = float(f1_score(y_true, y_pred, average="macro"))
            f2m = float(fbeta_score(y_true, y_pred, beta=2, average="micro"))

            return {
                "selected_model": "AutoML (AutoGluon)",
                "model_key": "automl",
                "problem_type": "classification",
                "preset": preset,
                "hpo_mode": "off" if hpo_mode in ("off", "none", "") else hpo_mode,
                "time_limit_s": int(time_limit_s),
                "seed": int(seed),
                "best_model_name": best_model_name,
                "metrics": {
                    "test_accuracy": acc,
                    "test_f1_macro": f1m,
                    "test_f2_micro": f2m,
                },
                "split_data": {
                    "train_table": train_df.to_dict(orient="records"),
                    "validation_table": val_df.to_dict(orient="records"),
                    "test_table": test_df.to_dict(orient="records"),
                },
            }

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--preset", default="best_quality")
    p.add_argument("--hpo_mode", default="off")
    p.add_argument("--time_limit_s", type=int, default=300)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_trials", type=int, default=20)
    p.add_argument("--out_json", required=True)
    p.add_argument("--artifact_path", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.data)

    result = train_autogluon(
        df=df,
        target_variable=args.label,
        preset=args.preset,
        hpo_mode=args.hpo_mode,
        time_limit_s=args.time_limit_s,
        seed=args.seed,
        num_trials=args.num_trials,
        artifact_path=args.artifact_path,
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
