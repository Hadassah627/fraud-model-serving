"""
api/schemas.py
Pydantic request and response schemas for FastAPI Model Serving.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    model_name: str = Field(..., example="E1_LightGBM")
    model_version: str = Field(..., example="E1")
    feature_count: int = Field(..., example=406)
    decision_threshold: float = Field(..., example=0.616521)


class TransactionRequest(BaseModel):
    """
    Raw transaction request schema.
    
    Accepts raw transaction fields (TransactionID, TransactionDT, TransactionAmt, etc.).
    Allows extra dynamic raw IEEE-CIS columns passed in JSON payload.
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: Optional[Union[int, str]] = Field(None, alias="TransactionID")
    TransactionDT: Optional[float] = Field(None, description="Transaction timestamp in seconds")
    TransactionAmt: Optional[float] = Field(None, description="Transaction amount")
    ProductCD: Optional[str] = Field(None, description="Product code")
    card1: Optional[Union[int, float]] = Field(None)
    card2: Optional[Union[int, float]] = Field(None)
    card3: Optional[Union[int, float]] = Field(None)
    card4: Optional[str] = Field(None)
    card5: Optional[Union[int, float]] = Field(None)
    card6: Optional[str] = Field(None)
    addr1: Optional[Union[int, float]] = Field(None)
    addr2: Optional[Union[int, float]] = Field(None)
    dist1: Optional[Union[int, float]] = Field(None)
    P_emaildomain: Optional[str] = Field(None)
    R_emaildomain: Optional[str] = Field(None)


class BatchTransactionRequest(BaseModel):
    """Batch transaction request schema."""
    transactions: List[TransactionRequest] = Field(..., min_items=1)


class PredictionResponse(BaseModel):
    """Fraud prediction response schema."""
    transaction_id: Union[int, str] = Field(..., example="3544193")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, example=0.003993)
    decision: str = Field(..., example="LEGIT")
    actual_label: Optional[str] = Field("UNKNOWN", example="LEGIT")
    preprocessing_time_ms: float = Field(..., example=0.594)
    model_prediction_time_ms: float = Field(..., example=0.042)
    total_inference_time_ms: float = Field(..., example=0.636)


class BatchPredictionResponse(BaseModel):
    """Batch fraud prediction response schema."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[Any] = None
