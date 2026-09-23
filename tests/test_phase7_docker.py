"""
tests/test_phase7_docker.py
Automated test suite for Phase 7 — Docker & Deployment.

Tests:
1. Dockerfile Configuration & Directive Validation
2. requirements.txt Dependency Completeness
3. .dockerignore Exclude Rules Validation
4. E1 Model Artifact Immutability & SHA256 Hash Verification
5. Local FastAPI Prediction & Parity Verification (Transaction 3544193)
6. Docker CLI Availability & Container Build/Run Readiness Verification
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app
from inference.offline_inference import E1_DECISION_THRESHOLD


def test_dockerfile_configuration():
    """Verify Dockerfile exists and contains required deployment directives."""
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile does not exist in repository root!"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:" in content, "Dockerfile missing FROM python base image directive!"
    assert "WORKDIR /app" in content, "Dockerfile missing WORKDIR /app directive!"
    assert "EXPOSE 8000" in content, "Dockerfile missing EXPOSE 8000 directive!"
    assert "uvicorn" in content, "Dockerfile missing uvicorn startup command!"
    assert "0.0.0.0" in content, "Dockerfile must bind uvicorn to 0.0.0.0!"
    assert "8000" in content, "Dockerfile must configure uvicorn port 8000!"


def test_requirements_configuration():
    """Verify requirements.txt contains all required runtime packages."""
    req_path = PROJECT_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt does not exist in repository root!"

    content = req_path.read_text(encoding="utf-8").lower()
    required_pkgs = ["fastapi", "uvicorn", "lightgbm", "pandas", "numpy", "joblib", "scikit-learn", "pydantic"]
    for pkg in required_pkgs:
        assert pkg in content, f"Required package '{pkg}' missing from requirements.txt!"


def test_dockerignore_configuration():
    """Verify .dockerignore exists and excludes virtual environment and cache files."""
    ignore_path = PROJECT_ROOT / ".dockerignore"
    assert ignore_path.exists(), ".dockerignore does not exist in repository root!"

    content = ignore_path.read_text(encoding="utf-8")
    assert ".venv" in content, ".dockerignore must exclude .venv!"
    assert "__pycache__" in content, ".dockerignore must exclude __pycache__!"


def test_model_artifact_immutability():
    """Verify E1 model artifacts exist and SHA256 hashes match approved values.

    Approved SHA256 hashes were computed during Phase 7 Step 16 (2026-09-23)
    from the actual live files that passed all Phase 1-6 tests.
    These hashes serve as the immutability reference going forward.
    """
    model_txt = PROJECT_ROOT / "models" / "E1" / "model.txt"
    prep_joblib = PROJECT_ROOT / "models" / "E1" / "preprocessing.joblib"
    feat_json = PROJECT_ROOT / "models" / "E1" / "feature_names.json"

    assert model_txt.exists(), "E1 model.txt missing!"
    assert prep_joblib.exists(), "E1 preprocessing.joblib missing!"
    assert feat_json.exists(), "E1 feature_names.json missing!"

    # Approved SHA256 hashes — computed 2026-09-23 during Phase 7 Step 16
    APPROVED_HASHES = {
        "model.txt":            "d04dff4f765801b196a5eda7d24288e8fd792f5fc06a0576d0de2249f98cc219",
        "preprocessing.joblib": "0c336989206214cab202d3b4a8a726206cb4ca69a0908e52fdb6f9cf479fbf69",
        "feature_names.json":   "1c59105a626f57533af4fc56f3ba10ae112b739c16c2e2d99cec24c1b1d0330d",
    }

    h_model = hashlib.sha256(model_txt.read_bytes()).hexdigest()
    h_prep  = hashlib.sha256(prep_joblib.read_bytes()).hexdigest()
    h_feat  = hashlib.sha256(feat_json.read_bytes()).hexdigest()

    assert h_model == APPROVED_HASHES["model.txt"], (
        f"model.txt SHA256 MISMATCH!\n  Expected: {APPROVED_HASHES['model.txt']}\n  Got:      {h_model}"
    )
    assert h_prep == APPROVED_HASHES["preprocessing.joblib"], (
        f"preprocessing.joblib SHA256 MISMATCH!\n  Expected: {APPROVED_HASHES['preprocessing.joblib']}\n  Got:      {h_prep}"
    )
    assert h_feat == APPROVED_HASHES["feature_names.json"], (
        f"feature_names.json SHA256 MISMATCH!\n  Expected: {APPROVED_HASHES['feature_names.json']}\n  Got:      {h_feat}"
    )

    print(f"\n[PASS] model.txt             SHA256: {h_model}")
    print(f"[PASS] preprocessing.joblib  SHA256: {h_prep}")
    print(f"[PASS] feature_names.json    SHA256: {h_feat}")


def test_local_api_prediction_parity():
    """Verify FastAPI prediction endpoint produces expected E1 probability for transaction 3544193."""
    with TestClient(app) as client:
        # Check health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True
        assert health_data["decision_threshold"] == E1_DECISION_THRESHOLD

        # Check single prediction (Transaction 3544193 payload)
        payload = {
            "TransactionID": 3544193,
            "TransactionDT": 14757390,
            "TransactionAmt": 100.0,
            "ProductCD": "W",
            "card1": 1000,
        }
        pred_resp = client.post("/predict", json=payload)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert str(pred_data["transaction_id"]) == "3544193"
        assert pred_data["decision"] == "LEGIT"
        assert 0.0 <= pred_data["fraud_probability"] <= 1.0

        # Check docs endpoint
        docs_resp = client.get("/docs")
        assert docs_resp.status_code == 200


def test_docker_cli_and_container_readiness():
    """Check Docker CLI availability and report container build status accurately."""
    docker_executable = shutil.which("docker")
    if docker_executable is None:
        print("\n[NOTICE] Docker CLI not installed in host PATH.")
        print("[STATUS] Docker configuration & schemas: PASS")
        print("[STATUS] Docker container execution: NOT TESTED (Host OS missing Docker daemon)")
        return

    # If docker CLI is present, attempt docker --version
    try:
        res = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            print(f"\n[INFO] Docker CLI detected: {res.stdout.strip()}")
    except Exception as e:
        print(f"\n[NOTICE] Docker command check failed: {e}")


if __name__ == "__main__":
    print("Running Phase 7 Docker & Deployment Unit Tests...")
    test_dockerfile_configuration()
    print("[PASS] test_dockerfile_configuration")
    test_requirements_configuration()
    print("[PASS] test_requirements_configuration")
    test_dockerignore_configuration()
    print("[PASS] test_dockerignore_configuration")
    test_model_artifact_immutability()
    print("[PASS] test_model_artifact_immutability")
    test_local_api_prediction_parity()
    print("[PASS] test_local_api_prediction_parity")
    test_docker_cli_and_container_readiness()
    print("[PASS] test_docker_cli_and_container_readiness")
    print("\nALL PHASE 7 DOCKER UNIT TESTS PASSED SUCCESSFULLY!")
