import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold, train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MAX_ONE_HOT_CATEGORIES = 20


class GPUEnvironmentError(RuntimeError):
    pass


GPU_RUNTIME: Optional[Dict[str, Any]] = None


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    elif hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cleanup_gpu(runtime: Dict[str, Any]) -> None:
    cp = runtime["cp"]
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def ensure_gpu_environment(device_id: int = 0, use_rmm_pool: bool = True) -> Dict[str, Any]:
    global GPU_RUNTIME
    if GPU_RUNTIME is not None:
        return GPU_RUNTIME

    try:
        import cupy as cp
    except Exception as exc:
        raise GPUEnvironmentError(f"No se pudo importar CuPy: {type(exc).__name__}: {exc}") from exc

    try:
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        driver_version = int(cp.cuda.runtime.driverGetVersion())
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise GPUEnvironmentError(
            f"No se pudo inicializar CUDA dentro de WSL2: {type(exc).__name__}: {exc}"
        ) from exc

    if driver_version <= 0 or device_count < 1:
        raise GPUEnvironmentError(
            "CUDA no ve una GPU utilizable en WSL2. "
            "Verifica el driver de Windows, `wsl --update`, `wsl --shutdown` y `/dev/dxg`."
        )

    try:
        cp.cuda.Device(device_id).use()
    except Exception as exc:
        raise GPUEnvironmentError(
            f"No se pudo seleccionar la GPU {device_id}: {type(exc).__name__}: {exc}"
        ) from exc

    runtime: Dict[str, Any] = {
        "cp": cp,
        "device_id": device_id,
        "runtime_version": runtime_version,
        "driver_version": driver_version,
    }

    try:
        import xgboost as xgb

        runtime["xgb"] = xgb
    except Exception as exc:
        raise GPUEnvironmentError(f"No se pudo importar XGBoost: {type(exc).__name__}: {exc}") from exc

    try:
        from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
        from cuml.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from cuml.svm import SVC
    except Exception as exc:
        raise GPUEnvironmentError(
            f"No se pudo importar cuML: {type(exc).__name__}: {exc}"
        ) from exc

    runtime.update(
        {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "ElasticNet": ElasticNet,
            "LogisticRegression": LogisticRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "RandomForestClassifier": RandomForestClassifier,
            "KNeighborsRegressor": KNeighborsRegressor,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SVC": SVC,
        }
    )

    if use_rmm_pool:
        try:
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator

            rmm.reinitialize(pool_allocator=True, managed_memory=False)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            runtime["rmm"] = rmm
        except Exception:
            runtime["rmm"] = None

    GPU_RUNTIME = runtime
    return runtime


def make_scenarios(
    ordinal_specs_nb1: Dict[str, List[str]],
    ordinal_specs_nb2: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    return [
        {"escenario": 1, "ordinal_specs": ordinal_specs_nb1, "scale_mode": "base", "binary_classification": False},
        {"escenario": 2, "ordinal_specs": ordinal_specs_nb2, "scale_mode": "base", "binary_classification": False},
        {"escenario": 3, "ordinal_specs": ordinal_specs_nb1, "scale_mode": "linear_10", "binary_classification": False},
        {"escenario": 4, "ordinal_specs": ordinal_specs_nb1, "scale_mode": "nonlinear_10", "binary_classification": False},
        {"escenario": 5, "ordinal_specs": ordinal_specs_nb2, "scale_mode": "linear_10", "binary_classification": False},
        {"escenario": 6, "ordinal_specs": ordinal_specs_nb2, "scale_mode": "nonlinear_10", "binary_classification": False},
        {"escenario": 7, "ordinal_specs": ordinal_specs_nb1, "scale_mode": "base", "binary_classification": True},
        {"escenario": 8, "ordinal_specs": ordinal_specs_nb2, "scale_mode": "base", "binary_classification": True},
    ]


def gpu_regression_model_names() -> List[str]:
    return [
        "LinearRegression",
        "Ridge",
        "ElasticNet",
        "RandomForestRegressor",
        "KNeighborsRegressor",
        "XGBoostRegressor",
    ]


def gpu_classification_model_names() -> List[str]:
    return [
        "LogisticRegression",
        "RandomForestClassifier",
        "KNeighborsClassifier",
        "SVC",
        "XGBoostClassifier",
    ]


def _ordinal_values(size: int, scale_mode: str) -> List[float]:
    if scale_mode == "base":
        return list(range(size))
    if scale_mode == "linear_10":
        return [float(10 * (idx + 1)) for idx in range(size)]
    if scale_mode == "nonlinear_10":
        values = []
        total = 0.0
        for idx in range(size):
            total += 10.0 * (idx + 1)
            values.append(total)
        return values
    raise ValueError(f"scale_mode no soportado: {scale_mode}")


@dataclass
class PreprocessorState:
    numeric_columns: List[str]
    numeric_medians: Dict[str, float]
    ordinal_columns: List[str]
    ordinal_maps: Dict[str, Dict[Any, float]]
    low_card_columns: List[str]
    low_card_levels: Dict[str, List[str]]
    high_card_columns: List[str]
    high_card_frequency_maps: Dict[str, Dict[str, float]]
    feature_columns: List[str]


class TabularGPUPreprocessor:
    def __init__(self, ordinal_specs: Dict[str, List[str]], scale_mode: str, max_one_hot_categories: int = MAX_ONE_HOT_CATEGORIES):
        self.ordinal_specs = ordinal_specs
        self.scale_mode = scale_mode
        self.max_one_hot_categories = max_one_hot_categories
        self.state: Optional[PreprocessorState] = None

    def fit(self, frame: pd.DataFrame) -> "TabularGPUPreprocessor":
        df = frame.copy()
        ordinal_columns = [col for col in self.ordinal_specs if col in df.columns]
        ordinal_maps = {
            col: dict(zip(self.ordinal_specs[col], _ordinal_values(len(self.ordinal_specs[col]), self.scale_mode)))
            for col in ordinal_columns
        }

        numeric_columns: List[str] = []
        for col in df.columns:
            if col in ordinal_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)

        categorical_candidates = [
            col
            for col in df.columns
            if col not in numeric_columns and col not in ordinal_columns
        ]

        low_card_columns: List[str] = []
        low_card_levels: Dict[str, List[str]] = {}
        high_card_columns: List[str] = []
        high_card_frequency_maps: Dict[str, Dict[str, float]] = {}

        for col in categorical_candidates:
            series = df[col].astype("string").fillna("Missing").astype(str)
            nunique = int(series.nunique(dropna=False))
            if nunique <= self.max_one_hot_categories:
                low_card_columns.append(col)
                low_card_levels[col] = sorted(series.unique().tolist())
            else:
                high_card_columns.append(col)
                freq = series.value_counts(normalize=True, dropna=False)
                high_card_frequency_maps[col] = {str(k): float(v) for k, v in freq.items()}

        numeric_medians = {}
        for col in numeric_columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            median = coerced.median()
            numeric_medians[col] = float(0.0 if pd.isna(median) else median)

        transformed = self._transform_internal(df, fit_mode=True, numeric_columns=numeric_columns, numeric_medians=numeric_medians, ordinal_columns=ordinal_columns, ordinal_maps=ordinal_maps, low_card_columns=low_card_columns, low_card_levels=low_card_levels, high_card_columns=high_card_columns, high_card_frequency_maps=high_card_frequency_maps)
        self.state = PreprocessorState(
            numeric_columns=numeric_columns,
            numeric_medians=numeric_medians,
            ordinal_columns=ordinal_columns,
            ordinal_maps=ordinal_maps,
            low_card_columns=low_card_columns,
            low_card_levels=low_card_levels,
            high_card_columns=high_card_columns,
            high_card_frequency_maps=high_card_frequency_maps,
            feature_columns=transformed.columns.tolist(),
        )
        return self

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("El preprocesador no ha sido ajustado.")

        transformed = self._transform_internal(
            frame.copy(),
            fit_mode=False,
            numeric_columns=self.state.numeric_columns,
            numeric_medians=self.state.numeric_medians,
            ordinal_columns=self.state.ordinal_columns,
            ordinal_maps=self.state.ordinal_maps,
            low_card_columns=self.state.low_card_columns,
            low_card_levels=self.state.low_card_levels,
            high_card_columns=self.state.high_card_columns,
            high_card_frequency_maps=self.state.high_card_frequency_maps,
        )
        return transformed.reindex(columns=self.state.feature_columns, fill_value=0.0)

    def _transform_internal(
        self,
        df: pd.DataFrame,
        *,
        fit_mode: bool,
        numeric_columns: List[str],
        numeric_medians: Dict[str, float],
        ordinal_columns: List[str],
        ordinal_maps: Dict[str, Dict[Any, float]],
        low_card_columns: List[str],
        low_card_levels: Dict[str, List[str]],
        high_card_columns: List[str],
        high_card_frequency_maps: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        blocks: List[pd.DataFrame] = []

        if numeric_columns:
            numeric_block = pd.DataFrame(index=df.index)
            for col in numeric_columns:
                numeric_block[col] = pd.to_numeric(df[col], errors="coerce").fillna(numeric_medians[col]).astype("float32")
            blocks.append(numeric_block)

        if ordinal_columns:
            ordinal_block = pd.DataFrame(index=df.index)
            for col in ordinal_columns:
                mapping = ordinal_maps[col]
                mapped = df[col].astype("string").fillna("Missing").astype(str).map(mapping)
                ordinal_block[col] = mapped.fillna(-1.0).astype("float32")
            blocks.append(ordinal_block)

        if low_card_columns:
            low_card_blocks: List[pd.DataFrame] = []
            for col in low_card_columns:
                series = (
                    df[col]
                    .astype("string")
                    .fillna("Missing")
                    .astype(str)
                    .astype(pd.CategoricalDtype(categories=low_card_levels[col]))
                )
                dummies = pd.get_dummies(series, prefix=col, dtype="float32")
                dummies = dummies.reindex(df.index, fill_value=0.0)
                low_card_blocks.append(dummies)
            if low_card_blocks:
                blocks.append(pd.concat(low_card_blocks, axis=1))

        if high_card_columns:
            high_card_block = pd.DataFrame(index=df.index)
            for col in high_card_columns:
                series = df[col].astype("string").fillna("Missing").astype(str)
                freq_map = high_card_frequency_maps[col]
                high_card_block[f"{col}__freq"] = series.map(freq_map).fillna(0.0).astype("float32")
            blocks.append(high_card_block)

        if not blocks:
            return pd.DataFrame(index=df.index)

        merged = pd.concat(blocks, axis=1)
        merged = merged.reindex(df.index, fill_value=0.0)
        merged = merged.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if fit_mode:
            merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged.astype("float32")


def prepare_target(y: pd.Series, *, task: str, binary: bool = False) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    if task == "regression":
        return pd.to_numeric(y, errors="coerce").astype("float32").to_numpy(), None

    y_series = y.astype("string").fillna("Missing").astype(str)
    if binary:
        y_series = y_series.replace(
            {
                "Deficiente": "Bajo",
                "Bajo": "Bajo",
                "Regular": "Bajo",
                "Alto": "Bueno",
                "Excelente": "Bueno",
            }
        )
    encoder = LabelEncoder()
    return encoder.fit_transform(y_series), encoder


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": _safe_rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


def _classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def get_regression_specs(runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
    xgb = runtime["xgb"]
    return [
        {
            "name": "LinearRegression",
            "builder": lambda params: runtime["LinearRegression"](),
            "grid": [{}],
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "Ridge",
            "builder": lambda params: runtime["Ridge"](alpha=params["alpha"]),
            "grid": [{"alpha": value} for value in [0.1, 1.0, 10.0]],
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "ElasticNet",
            "builder": lambda params: runtime["ElasticNet"](alpha=params["alpha"], l1_ratio=params["l1_ratio"]),
            "grid": list(ParameterGrid({"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]})),
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "RandomForestRegressor",
            "builder": lambda params: runtime["RandomForestRegressor"](
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                random_state=RANDOM_STATE,
                n_streams=1,
            ),
            "grid": list(ParameterGrid({"n_estimators": [200, 400], "max_depth": [12, 20], "max_features": [0.7, 1.0]})),
            "needs_scaling": False,
            "backend": "cuml",
        },
        {
            "name": "KNeighborsRegressor",
            "builder": lambda params: runtime["KNeighborsRegressor"](n_neighbors=params["n_neighbors"]),
            "grid": [{"n_neighbors": value} for value in [5, 11, 21]],
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "XGBoostRegressor",
            "builder": lambda params: xgb.XGBRegressor(
                objective="reg:squarederror",
                device="cuda",
                tree_method="hist",
                random_state=RANDOM_STATE,
                eval_metric="mae",
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
            ),
            "grid": list(
                ParameterGrid(
                    {
                        "n_estimators": [300, 500],
                        "max_depth": [4, 8],
                        "learning_rate": [0.05, 0.10],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8],
                    }
                )
            ),
            "needs_scaling": False,
            "backend": "xgboost",
        },
    ]


def get_classification_specs(runtime: Dict[str, Any], binary: bool, num_classes: int) -> List[Dict[str, Any]]:
    xgb = runtime["xgb"]
    xgb_objective = "binary:logistic" if binary else "multi:softmax"
    xgb_eval_metric = "logloss" if binary else "mlogloss"

    return [
        {
            "name": "LogisticRegression",
            "builder": lambda params: runtime["LogisticRegression"](C=params["C"], max_iter=1000),
            "grid": [{"C": value} for value in [0.5, 1.0, 2.0]],
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "RandomForestClassifier",
            "builder": lambda params: runtime["RandomForestClassifier"](
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                random_state=RANDOM_STATE,
                n_streams=1,
            ),
            "grid": list(ParameterGrid({"n_estimators": [200, 400], "max_depth": [12, 20], "max_features": [0.7, 1.0]})),
            "needs_scaling": False,
            "backend": "cuml",
        },
        {
            "name": "KNeighborsClassifier",
            "builder": lambda params: runtime["KNeighborsClassifier"](n_neighbors=params["n_neighbors"]),
            "grid": [{"n_neighbors": value} for value in [5, 11, 21]],
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "SVC",
            "builder": lambda params: runtime["SVC"](C=params["C"], kernel="rbf", gamma=params["gamma"]),
            "grid": list(ParameterGrid({"C": [1.0, 5.0], "gamma": ["scale"]})),
            "needs_scaling": True,
            "backend": "cuml",
        },
        {
            "name": "XGBoostClassifier",
            "builder": lambda params: xgb.XGBClassifier(
                objective=xgb_objective,
                num_class=None if binary else num_classes,
                device="cuda",
                tree_method="hist",
                random_state=RANDOM_STATE,
                eval_metric=xgb_eval_metric,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
            ),
            "grid": list(
                ParameterGrid(
                    {
                        "n_estimators": [300, 500],
                        "max_depth": [4, 8],
                        "learning_rate": [0.05, 0.10],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8],
                    }
                )
            ),
            "needs_scaling": False,
            "backend": "xgboost",
        },
    ]


def _fit_and_predict(runtime: Dict[str, Any], spec: Dict[str, Any], params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> Tuple[np.ndarray, float, float]:
    model = spec["builder"](params)
    cp = runtime["cp"]
    start_train = time.time()
    try:
        if spec["backend"] == "cuml":
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train)
            model.fit(X_train_gpu, y_train_gpu)
            train_time = time.time() - start_train

            start_pred = time.time()
            X_pred_gpu = cp.asarray(X_pred, dtype=cp.float32)
            pred = model.predict(X_pred_gpu)
            pred_time = time.time() - start_pred

            return _to_numpy(pred), train_time, pred_time

        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        pred = model.predict(X_pred)
        pred_time = time.time() - start_pred

        return _to_numpy(pred), train_time, pred_time
    finally:
        try:
            del model
        except Exception:
            pass
        _cleanup_gpu(runtime)


def _cv_and_test_for_model(
    *,
    runtime: Dict[str, Any],
    spec: Dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_col: str,
    ordinal_specs: Dict[str, List[str]],
    scale_mode: str,
    task: str,
    binary: bool = False,
) -> Dict[str, Any]:
    y_train_raw = train_frame[target_col]
    y_test_raw = test_frame[target_col]
    y_train, encoder = prepare_target(y_train_raw, task=task, binary=binary)
    y_test, _ = prepare_target(y_test_raw, task=task, binary=binary)

    splitter = (
        StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if task == "classification"
        else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    )
    objective_name = "MAPE_%" if task == "regression" else "F1_macro"
    objective_direction = "min" if task == "regression" else "max"

    best_result: Optional[Dict[str, Any]] = None

    for params in spec["grid"]:
        fold_rows: List[Dict[str, Any]] = []
        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(train_frame, y_train), start=1):
            fold_train = (
                train_frame.iloc[train_idx]
                .drop(columns=[target_col], errors="ignore")
                .copy()
            )
            fold_val = (
                train_frame.iloc[val_idx]
                .drop(columns=[target_col], errors="ignore")
                .copy()
            )
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]

            preprocessor = TabularGPUPreprocessor(ordinal_specs=ordinal_specs, scale_mode=scale_mode)
            X_fold_train = preprocessor.fit_transform(fold_train).to_numpy(dtype=np.float32, copy=False)
            X_fold_val = preprocessor.transform(fold_val).to_numpy(dtype=np.float32, copy=False)

            if X_fold_train.shape[0] != len(y_fold_train) or X_fold_val.shape[0] != len(y_fold_val):
                raise ValueError(
                    "El preprocesamiento altero el numero de filas en validacion cruzada. "
                    f"Fold={fold_id}, X_train={X_fold_train.shape[0]}, y_train={len(y_fold_train)}, "
                    f"X_val={X_fold_val.shape[0]}, y_val={len(y_fold_val)}."
                )

            if spec["needs_scaling"]:
                scaler = StandardScaler()
                X_fold_train = scaler.fit_transform(X_fold_train).astype(np.float32, copy=False)
                X_fold_val = scaler.transform(X_fold_val).astype(np.float32, copy=False)

            y_pred, train_time, pred_time = _fit_and_predict(runtime, spec, params, X_fold_train, y_fold_train, X_fold_val)
            metrics = _regression_scores(y_fold_val, y_pred) if task == "regression" else _classification_scores(y_fold_val, y_pred)
            metrics["Fold"] = fold_id
            metrics["Fit_Time"] = train_time
            metrics["Pred_Time"] = pred_time
            fold_rows.append(metrics)

            del preprocessor, X_fold_train, X_fold_val, y_pred
            _cleanup_gpu(runtime)

        folds_df = pd.DataFrame(fold_rows)
        summary_row = {
            f"CV_{col}_Mean": float(folds_df[col].mean())
            for col in folds_df.columns
            if col != "Fold"
        }
        summary_row.update(
            {
                f"CV_{col}_Std": float(folds_df[col].std(ddof=0))
                for col in folds_df.columns
                if col != "Fold"
            }
        )
        summary_row["params"] = params
        score = summary_row[f"CV_{objective_name}_Mean"]

        if best_result is None:
            best_result = {"params": params, "cv_folds": folds_df, "cv_summary": pd.DataFrame([summary_row]), "score": score}
        else:
            is_better = score < best_result["score"] if objective_direction == "min" else score > best_result["score"]
            if is_better:
                best_result = {"params": params, "cv_folds": folds_df, "cv_summary": pd.DataFrame([summary_row]), "score": score}

    assert best_result is not None

    train_features = train_frame.drop(columns=[target_col], errors="ignore").copy()
    test_features = test_frame.drop(columns=[target_col], errors="ignore").copy()
    preprocessor = TabularGPUPreprocessor(ordinal_specs=ordinal_specs, scale_mode=scale_mode)
    X_train = preprocessor.fit_transform(train_features).to_numpy(dtype=np.float32, copy=False)
    X_test = preprocessor.transform(test_features).to_numpy(dtype=np.float32, copy=False)

    if X_train.shape[0] != len(y_train) or X_test.shape[0] != len(y_test):
        raise ValueError(
            "El preprocesamiento altero el numero de filas en entrenamiento/prueba. "
            f"X_train={X_train.shape[0]}, y_train={len(y_train)}, "
            f"X_test={X_test.shape[0]}, y_test={len(y_test)}."
        )

    if spec["needs_scaling"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    y_pred_test, test_fit_time, test_pred_time = _fit_and_predict(runtime, spec, best_result["params"], X_train, y_train, X_test)
    test_metrics = _regression_scores(y_test, y_pred_test) if task == "regression" else _classification_scores(y_test, y_pred_test)
    test_metrics["Fit_Time"] = test_fit_time
    test_metrics["Pred_Time"] = test_pred_time

    result = {
        "best_params": best_result["params"],
        "cv_folds": best_result["cv_folds"],
        "cv_summary": best_result["cv_summary"],
        "test_metrics": test_metrics,
    }
    if encoder is not None:
        result["label_classes"] = encoder.classes_.tolist()

    del preprocessor, X_train, X_test, y_pred_test
    _cleanup_gpu(runtime)
    return result


def fit_model_for_saving(
    runtime: Dict[str, Any],
    spec: Dict[str, Any],
    params: Dict[str, Any],
    train_frame: pd.DataFrame,
    target_col: str,
    ordinal_specs: Dict[str, List[str]],
    scale_mode: str,
    task: str,
    binary: bool = False,
) -> Dict[str, Any]:
    """Colabora con el guardado del modelo final entrenado en todo el set de entrenamiento."""
    y_train_raw = train_frame[target_col]
    y_train, encoder = prepare_target(y_train_raw, task=task, binary=binary)

    preprocessor = TabularGPUPreprocessor(ordinal_specs=ordinal_specs, scale_mode=scale_mode)
    X_train = preprocessor.fit_transform(train_frame.drop(columns=[target_col], errors="ignore")).to_numpy(dtype=np.float32, copy=False)

    scaler = None
    if spec["needs_scaling"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)

    # Entrenar modelo final
    model = spec["builder"](params)
    cp = runtime["cp"]
    if spec["backend"] == "cuml":
        X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train)
        model.fit(X_train_gpu, y_train_gpu)
    else:
        model.fit(X_train, y_train)

    _cleanup_gpu(runtime)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "encoder": encoder,
    }


def evaluate_regression_models(
    df_model: pd.DataFrame,
    target_col: str,
    ordinal_specs: Dict[str, List[str]],
    scale_mode: str,
    top_n: int = 5,
    model_tag: str = "Regression",
) -> Dict[str, Any]:
    runtime = ensure_gpu_environment()

    # Intento de auto-tagging si el usuario no paso uno especifico
    if model_tag == "Regression":
        if "saber11" in target_col.lower():
            model_tag = "Saber11"
        elif "saberpro" in target_col.lower():
            model_tag = "SaberPro"

    frame = df_model.dropna(subset=[target_col]).copy()
    train_frame, test_frame = train_test_split(frame, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    tuned_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for spec in get_regression_specs(runtime):
        result = _cv_and_test_for_model(
            runtime=runtime,
            spec=spec,
            train_frame=train_frame,
            test_frame=test_frame,
            target_col=target_col,
            ordinal_specs=ordinal_specs,
            scale_mode=scale_mode,
            task="regression",
        )
        tuned_results[spec["name"]] = result
        row = {
            "Modelo": spec["name"],
            "Best_Params": str(result["best_params"]),
            "Test_MAE": result["test_metrics"]["MAE"],
            "Test_RMSE": result["test_metrics"]["RMSE"],
            "Test_R2": result["test_metrics"]["R2"],
            "Test_MAPE_%": result["test_metrics"]["MAPE_%"],
            "CV_MAE_Mean": float(result["cv_summary"]["CV_MAE_Mean"].iloc[0]),
            "CV_RMSE_Mean": float(result["cv_summary"]["CV_RMSE_Mean"].iloc[0]),
            "CV_R2_Mean": float(result["cv_summary"]["CV_R2_Mean"].iloc[0]),
            "CV_MAPE_%_Mean": float(result["cv_summary"]["CV_MAPE_%_Mean"].iloc[0]),
            "Fit_Time_Avg": float(result["cv_summary"]["CV_Fit_Time_Mean"].iloc[0]),
            "Pred_Time_Avg": float(result["cv_summary"]["CV_Pred_Time_Mean"].iloc[0]),
        }
        # Overfit detection: Difference between CV and Test MAPE
        row["Overfit_Diff"] = abs(row["Test_MAPE_%"] - row["CV_MAPE_%_Mean"])
        summary_rows.append(row)

    tuned_summary = pd.DataFrame(summary_rows).sort_values(by="Test_MAPE_%", ascending=True).head(top_n).reset_index(drop=True)
    tuned_results = {name: tuned_results[name] for name in tuned_summary["Modelo"].tolist()}

    # Guardar el mejor modelo
    if not tuned_summary.empty:
        best_row = tuned_summary.iloc[0]
        winner_name = best_row["Modelo"]
        winner_spec = next(s for s in get_regression_specs(runtime) if s["name"] == winner_name)
        winner_params = tuned_results[winner_name]["best_params"]

        print(f"--- Guardando el mejor modelo de Regresion para {model_tag}: {winner_name} ---")
        best_bundle = fit_model_for_saving(
            runtime=runtime,
            spec=winner_spec,
            params=winner_params,
            train_frame=train_frame,
            target_col=target_col,
            ordinal_specs=ordinal_specs,
            scale_mode=scale_mode,
            task="regression"
        )
        # Use a clean filename based on the tag
        clean_tag = model_tag.lower().replace(" ", "_")
        save_path = f"best_model_{clean_tag}_regression.joblib"
        joblib.dump(best_bundle, save_path)
        print(f"Modelo guardado como: {save_path}")

    return {"tuned_results": tuned_results, "tuned_summary": tuned_summary}


def evaluate_classification_models(
    df_model: pd.DataFrame,
    target_col: str,
    ordinal_specs: Dict[str, List[str]],
    scale_mode: str,
    top_n: int = 5,
    binary: bool = False,
    model_tag: str = "Classification",
) -> Dict[str, Any]:
    runtime = ensure_gpu_environment()

    # Intento de auto-tagging si el usuario no paso uno especifico
    if model_tag == "Classification":
        if "saber11" in target_col.lower():
            model_tag = "Saber11"
        elif "saberpro" in target_col.lower():
            model_tag = "SaberPro"

    frame = df_model.dropna(subset=[target_col]).copy()
    y_all, _ = prepare_target(frame[target_col], task="classification", binary=binary)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    y_train_preview, encoder = prepare_target(train_frame[target_col], task="classification", binary=binary)
    num_classes = int(len(np.unique(y_train_preview)))
    del y_train_preview, encoder

    tuned_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for spec in get_classification_specs(runtime, binary=binary, num_classes=num_classes):
        result = _cv_and_test_for_model(
            runtime=runtime,
            spec=spec,
            train_frame=train_frame,
            test_frame=test_frame,
            target_col=target_col,
            ordinal_specs=ordinal_specs,
            scale_mode=scale_mode,
            task="classification",
            binary=binary,
        )
        tuned_results[spec["name"]] = result
        row = {
            "Modelo": spec["name"],
            "Best_Params": str(result["best_params"]),
            "Test_Accuracy": result["test_metrics"]["Accuracy"],
            "Test_Balanced_Accuracy": result["test_metrics"]["Balanced_Accuracy"],
            "Test_Precision_macro": result["test_metrics"]["Precision_macro"],
            "Test_Recall_macro": result["test_metrics"]["Recall_macro"],
            "Test_F1_macro": result["test_metrics"]["F1_macro"],
            "CV_Accuracy_Mean": float(result["cv_summary"]["CV_Accuracy_Mean"].iloc[0]),
            "CV_Balanced_Accuracy_Mean": float(result["cv_summary"]["CV_Balanced_Accuracy_Mean"].iloc[0]),
            "CV_Precision_macro_Mean": float(result["cv_summary"]["CV_Precision_macro_Mean"].iloc[0]),
            "CV_Recall_macro_Mean": float(result["cv_summary"]["CV_Recall_macro_Mean"].iloc[0]),
            "CV_F1_macro_Mean": float(result["cv_summary"]["CV_F1_macro_Mean"].iloc[0]),
            "Fit_Time_Avg": float(result["cv_summary"]["CV_Fit_Time_Mean"].iloc[0]),
            "Pred_Time_Avg": float(result["cv_summary"]["CV_Pred_Time_Mean"].iloc[0]),
        }
        # Overfit detection: Difference between CV and Test F1 (score, so higher is better, lower diff is better)
        row["Overfit_Diff"] = abs(row["Test_F1_macro"] - row["CV_F1_macro_Mean"])
        summary_rows.append(row)

    tuned_summary = pd.DataFrame(summary_rows).sort_values(by="Test_F1_macro", ascending=False).head(top_n).reset_index(drop=True)
    tuned_results = {name: tuned_results[name] for name in tuned_summary["Modelo"].tolist()}

    # Guardar el mejor modelo
    if not tuned_summary.empty:
        best_row = tuned_summary.iloc[0]
        winner_name = best_row["Modelo"]
        winner_spec = next(s for s in get_classification_specs(runtime, binary, num_classes) if s["name"] == winner_name)
        winner_params = tuned_results[winner_name]["best_params"]

        print(f"--- Guardando el mejor modelo de Clasificación para {model_tag}: {winner_name} ---")
        best_bundle = fit_model_for_saving(
            runtime=runtime,
            spec=winner_spec,
            params=winner_params,
            train_frame=train_frame,
            target_col=target_col,
            ordinal_specs=ordinal_specs,
            scale_mode=scale_mode,
            task="classification",
            binary=binary
        )
        # Use a clean filename based on the tag
        clean_tag = model_tag.lower().replace(" ", "_")
        save_path = f"best_model_{clean_tag}_classification.joblib"
        joblib.dump(best_bundle, save_path)
        print(f"Modelo guardado como: {save_path}")

    return {"tuned_results": tuned_results, "tuned_summary": tuned_summary}
