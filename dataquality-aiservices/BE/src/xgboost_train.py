import time
import numpy as np

from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
    ParameterGrid,
    ParameterSampler,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _infer_task_type(y, max_classes_for_classification: int = 20) -> str:
    y_values = y.dropna()

    if str(y_values.dtype) in ("object", "bool", "category"):
        return "classification"

    n_unique = int(y_values.nunique(dropna=True))
    n = int(len(y_values))

    if np.issubdtype(y_values.dtype, np.integer) and n_unique <= max_classes_for_classification:
        return "classification"

    if n_unique <= max_classes_for_classification and (n_unique / max(1, n)) < 0.05:
        return "classification"

    return "regression"


def _can_stratify(y) -> bool:
    try:
        vc = y.value_counts(dropna=False)
        return int(vc.min()) >= 2
    except Exception:
        _, counts = np.unique(np.asarray(y), return_counts=True)
        return int(counts.min()) >= 2


def train_xgboost(
    df_encoded,
    target_variable: str,
    MODE: str = "none",
    SCORING: str = "accuracy",
    N_SPLITS: int = 5,
    N_ITER: int = 25,
    use_cross_validation: bool = False,
    time_limit_s: int = 60,
    seed: int | None = None,
    task_type: str | None = None,
    early_stopping_rounds: int = 50,
):
    TARGET_VARIABLE = target_variable
    if TARGET_VARIABLE not in df_encoded.columns:
        raise ValueError(f"Target '{TARGET_VARIABLE}' not in df columns.")

    X = df_encoded.drop(columns=[TARGET_VARIABLE])
    y = df_encoded[TARGET_VARIABLE]

    non_numeric_cols = list(X.select_dtypes(exclude=["number"]).columns)
    if non_numeric_cols:
        raise ValueError(
            f"df_encoded must be fully numeric after preprocessing. Non-numeric columns: {non_numeric_cols[:20]}"
            + (" ..." if len(non_numeric_cols) > 20 else "")
        )

    if task_type is None:
        task_type = _infer_task_type(y)

    if task_type not in ("classification", "regression"):
        raise ValueError('task_type must be "classification" or "regression" (or None for auto).')

    if task_type == "classification":
        if SCORING not in ("accuracy",):
            SCORING = "accuracy"
    else:
        SCORING = "mae"

    strat = None
    if task_type == "classification" and _can_stratify(y):
        strat = y

    X_train_full, X_test_data, y_train_full, y_test_data = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=strat,
    )

    strat_train = None
    if task_type == "classification" and _can_stratify(y_train_full):
        strat_train = y_train_full

    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
        stratify=strat_train,
    )

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    if task_type == "classification":
        ModelCls = XGBClassifier
        base_model_kwargs = dict(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            eval_metric="error",
            use_label_encoder=False,
        )

        fixed_params = {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        }

        param_grid = {
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5, 15],
            "learning_rate": [0.02, 0.05, 0.1],
            "n_estimators": [300, 700, 1200],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.5],
            "reg_lambda": [0.1, 1.0, 10.0],
            "reg_alpha": [0.0, 1e-3, 0.1],
        }

        param_dist = {
            "n_estimators": randint(200, 2001),
            "max_depth": randint(2, 11),
            "learning_rate": loguniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": loguniform(0.5, 32.0),
            "gamma": loguniform(1e-4, 5.0),
            "reg_lambda": loguniform(1e-3, 50.0),
            "reg_alpha": loguniform(1e-6, 5.0),
        }

    else:
        ModelCls = XGBRegressor
        base_model_kwargs = dict(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            eval_metric="mae",
        )

        fixed_params = {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        }

        param_grid = {
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5, 15],
            "learning_rate": [0.02, 0.05, 0.1],
            "n_estimators": [300, 700, 1200],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.5],
            "reg_lambda": [0.1, 1.0, 10.0],
            "reg_alpha": [0.0, 1e-3, 0.1],
        }

        param_dist = {
            "n_estimators": randint(200, 2001),
            "max_depth": randint(2, 11),
            "learning_rate": loguniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": loguniform(0.5, 32.0),
            "gamma": loguniform(1e-4, 5.0),
            "reg_lambda": loguniform(1e-3, 50.0),
            "reg_alpha": loguniform(1e-6, 5.0),
        }

    def _selection_score(y_true, y_pred):
        if task_type == "classification":
            return float(accuracy_score(y_true, y_pred))
        return float(-mean_absolute_error(y_true, y_pred))

    if task_type == "classification":
        sklearn_selection_scoring = "accuracy"
    else:
        sklearn_selection_scoring = "neg_mean_absolute_error"

    def _fit_with_optional_early_stopping(model, X_tr, y_tr):
        if use_cross_validation:
            model.fit(X_tr, y_tr)
            return model
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val_data, y_val_data)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model

    t0 = time.time()

    def time_left():
        return (time.time() - t0) < max(1, int(time_limit_s))

    best_val_score = -np.inf
    best_params = None
    best_model = None

    if MODE == "none":
        model = ModelCls(**fixed_params, **base_model_kwargs)

        if use_cross_validation:
            cv_scores = cross_val_score(
                model,
                X_train_full,
                y_train_full,
                cv=cv,
                scoring=sklearn_selection_scoring,
                n_jobs=-1,
            )
            best_val_score = float(np.mean(cv_scores))
            model.fit(X_train_full, y_train_full)
        else:
            model = _fit_with_optional_early_stopping(model, X_train_data, y_train_data)
            y_val_pred = model.predict(X_val_data)
            best_val_score = _selection_score(y_val_data, y_val_pred)

        best_model = model
        best_params = fixed_params

    elif MODE == "grid":
        for params in ParameterGrid(param_grid):
            if not time_left():
                break

            m = ModelCls(**params, **base_model_kwargs)

            if use_cross_validation:
                cv_scores = cross_val_score(
                    m,
                    X_train_full,
                    y_train_full,
                    cv=cv,
                    scoring=sklearn_selection_scoring,
                    n_jobs=-1,
                )
                val_score = float(np.mean(cv_scores))
            else:
                m = _fit_with_optional_early_stopping(m, X_train_data, y_train_data)
                y_val_pred = m.predict(X_val_data)
                val_score = _selection_score(y_val_data, y_val_pred)

            if best_params is None or val_score > best_val_score:
                best_val_score = val_score
                best_params = params

        if best_params is None:
            raise RuntimeError("Grid Search did not evaluate any configuration (time limit too small).")

        best_model = ModelCls(**best_params, **base_model_kwargs)
        if use_cross_validation:
            best_model.fit(X_train_full, y_train_full)
        else:
            best_model = _fit_with_optional_early_stopping(best_model, X_train_data, y_train_data)

    elif MODE == "random":
        sampler = ParameterSampler(param_dist, n_iter=N_ITER, random_state=seed)

        for params in sampler:
            if not time_left():
                break

            m = ModelCls(**params, **base_model_kwargs)

            if use_cross_validation:
                cv_scores = cross_val_score(
                    m,
                    X_train_full,
                    y_train_full,
                    cv=cv,
                    scoring=sklearn_selection_scoring,
                    n_jobs=-1,
                )
                val_score = float(np.mean(cv_scores))
            else:
                m = _fit_with_optional_early_stopping(m, X_train_data, y_train_data)
                y_val_pred = m.predict(X_val_data)
                val_score = _selection_score(y_val_data, y_val_pred)

            if best_params is None or val_score > best_val_score:
                best_val_score = val_score
                best_params = params

        if best_params is None:
            raise RuntimeError("Random Search did not evaluate any configuration (time limit too small).")

        best_model = ModelCls(**best_params, **base_model_kwargs)
        if use_cross_validation:
            best_model.fit(X_train_full, y_train_full)
        else:
            best_model = _fit_with_optional_early_stopping(best_model, X_train_data, y_train_data)

    else:
        raise ValueError('MODE must be "none", "grid", or "random"')

    y_pred_test = best_model.predict(X_test_data)

    if task_type == "classification":
        acc = float(accuracy_score(y_test_data, y_pred_test))
        f1m = float(f1_score(y_test_data, y_pred_test, average="macro"))
        f2m = float(fbeta_score(y_test_data, y_pred_test, beta=2, average="micro"))
        metrics = {
            "test_accuracy": acc,
            "test_f1_macro": f1m,
            "test_f2_micro": f2m,
        }
        best_val_score_out = float(best_val_score)
    else:
        mae = float(mean_absolute_error(y_test_data, y_pred_test))
        rmse = float(mean_squared_error(y_test_data, y_pred_test, squared=False))
        r2 = float(r2_score(y_test_data, y_pred_test))
        metrics = {
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
        }
        best_val_score_out = float(-best_val_score)

    p = best_model.get_params()
    used = {k: p.get(k) for k in [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_lambda",
        "reg_alpha",
        "tree_method",
    ]}

    best_iter = getattr(best_model, "best_iteration", None)
    if best_iter is not None:
        used["best_iteration"] = int(best_iter)
        used["best_ntree_limit"] = int(getattr(best_model, "best_ntree_limit", best_iter + 1))

    return {
        "model": "XGBoost",
        "model_key": "xgb",
        "task_type": task_type,
        "search_mode": MODE,
        "cross_validation": bool(use_cross_validation),
        "main_scoring": ("accuracy" if task_type == "classification" else "mae"),
        "time_limit_s": int(time_limit_s),
        "seed": int(seed),
        "hyperparameters": used,
        "best_val_score": best_val_score_out,
        "metrics": metrics,
        "_model": best_model,
        "split_indices": {
            "train_idx": X_train_data.index.tolist(),
            "validation_idx": X_val_data.index.tolist(),
            "test_idx": X_test_data.index.tolist(),
        },
    }