"""
monitoring/model_metadata.py
Phase 8 — Single source of truth for E1 model identity and configuration.

This module is the canonical reference for all model version information.
api/main.py, health endpoints, and inference loggers all read from here —
no values are duplicated or hardcoded elsewhere.

IMPORTANT: Do NOT modify E1_DECISION_THRESHOLD here directly.
It is imported from inference.offline_inference to avoid duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so this module is importable standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.offline_inference import E1_DECISION_THRESHOLD

# ── E1 Model Metadata (Single Source of Truth) ───────────────────────────────
#
# model_version:  Short version tag used in logs and monitoring records.
# model_name:     Full name used in /health responses and experiment records.
# All artifact paths are relative to the project root (same as api/main.py).
#
MODEL_METADATA: dict = {
    "model_version":        "E1",
    "model_name":           "E1_LightGBM",
    "feature_count":        406,
    "decision_threshold":   E1_DECISION_THRESHOLD,   # 0.616521 — imported, not duplicated
    "artifact_model":       "models/E1/model.txt",
    "artifact_preprocessor":"models/E1/preprocessing.joblib",
    "artifact_features":    "models/E1/feature_names.json",
    "source_repository":    "adaptive-upi-fraud-detection",
    "serving_repository":   "fraud-model-serving",
}
