"""
src/features/ieee_cis_features.py
Feature engineering and leakage-safe preprocessing for IEEE-CIS.
Copied for self-contained model serving deserialization of preprocessing.joblib.
"""
from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Column definitions ───────────────────────────────────────────────────────

SAFE_D_COLS = ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]
DROP_D_COLS = ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]
DROP_OTHER = ["dist2"]
ALWAYS_DROP = ["isFraud", "TransactionID", "TransactionDT"]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

V_COL_MISS_THRESHOLD = 0.80

MISSINGNESS_INDICATOR_COLS = [
    "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain",
    "card2", "card3", "card5",
    "DeviceInfo", "DeviceType",
    "id_03", "id_04", "id_05", "id_06", "id_09", "id_10",
    "id_14", "id_18", "id_30", "id_33",
]


class IEEECISPreprocessor:
    """
    Leakage-safe preprocessing pipeline for IEEE-CIS.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._is_fitted = False

        self._drop_cols: list[str] = []
        self._v_drop_cols: list[str] = []
        self._num_medians: dict[str, float] = {}
        self._cat_encoders: dict[str, LabelEncoder] = {}
        self._feature_cols: list[str] = []
        self._leakage_log: list[dict] = []

    def fit(self, df_train: pd.DataFrame) -> "IEEECISPreprocessor":
        logger.info("Fitting preprocessor on training data (%d rows) ...", len(df_train))
        v_threshold = self.cfg.get("features", {}).get("v_column_drop_missing_threshold", 0.80)
        all_v = [c for c in df_train.columns if c.startswith("V")]
        self._v_drop_cols = [
            c for c in all_v
            if df_train[c].isnull().mean() > v_threshold
        ]
        for c in self._v_drop_cols:
            self._leakage_log.append({
                "column": c, "action": "DROP",
                "reason": f">={v_threshold*100:.0f}% missing in training data",
            })

        drop_set = (
            set(ALWAYS_DROP) | set(DROP_D_COLS) | set(DROP_OTHER) |
            set(self._v_drop_cols) | set(CATEGORICAL_COLS)
        )
        num_cols = [
            c for c in df_train.columns
            if c not in drop_set and df_train[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.int16)
        ]
        for c in num_cols:
            median_val = df_train[c].median()
            self._num_medians[c] = float(median_val) if not math.isnan(median_val) else 0.0

        for c in CATEGORICAL_COLS:
            if c not in df_train.columns:
                continue
            le = LabelEncoder()
            series = df_train[c].fillna("MISSING").astype(str)
            le.fit(series)
            extra = []
            if "MISSING" not in le.classes_:
                extra.append("MISSING")
            if "UNSEEN" not in le.classes_:
                extra.append("UNSEEN")
            if extra:
                le.classes_ = np.concatenate([le.classes_, extra])
            self._cat_encoders[c] = le

        self._feature_cols = self._build_feature_list(df_train)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")

        df = df.copy()
        y = df["isFraud"].astype(int) if "isFraud" in df.columns else pd.Series(0, index=df.index)

        if self.cfg.get("features", {}).get("create_dt_features", True):
            df = _create_temporal_features(df)

        if self.cfg.get("features", {}).get("create_missingness_indicators", True):
            df = _create_missingness_indicators(df, MISSINGNESS_INDICATOR_COLS)

        for c, med in self._num_medians.items():
            if c in df.columns:
                df[c] = df[c].fillna(med)

        for c, le in self._cat_encoders.items():
            if c not in df.columns:
                continue
            series = df[c].fillna("MISSING").astype(str)
            known = set(le.classes_)
            series = series.apply(lambda x: x if x in known else "UNSEEN")
            df[c] = le.transform(series)

        avail = [c for c in self._feature_cols if c in df.columns]
        X = df[avail].copy()

        return X, y

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IEEECISPreprocessor":
        return joblib.load(Path(path))

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    @property
    def leakage_log(self) -> list[dict]:
        return self._leakage_log

    def _build_feature_list(self, df: pd.DataFrame) -> list[str]:
        exclude = (
            set(ALWAYS_DROP) |
            set(DROP_D_COLS) |
            set(DROP_OTHER) |
            set(self._v_drop_cols)
        )
        temporal = ["hour_sin", "hour_cos", "day_index"]
        miss_names = [f"{c}_missing" for c in MISSINGNESS_INDICATOR_COLS]
        orig = [c for c in df.columns if c not in exclude]

        all_cols = orig + temporal + miss_names
        seen = set()
        result = []
        for c in all_cols:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result


def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["TransactionDT"]
    dt_min = 86400.0
    hour = ((dt % 86400) / 3600).astype(float)
    df["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    df["day_index"] = ((dt - dt_min) / 86400).astype(float)
    return df


def _create_missingness_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isnull().astype(np.int8)
    return df
