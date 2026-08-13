"""Pydantic schemas for the Fraud Detection API request/response contract."""

from typing import Literal

from pydantic import BaseModel, Field

CountryCode = Literal[
    "AR",
    "AU",
    "BR",
    "CA",
    "CH",
    "CL",
    "CO",
    "ES",
    "FR",
    "GB",
    "GT",
    "IT",
    "KR",
    "MX",
    "PT",
    "TR",
    "UA",
    "US",
    "UY",
]


class Transaction(BaseModel):
    """Input transaction, as received from the client — before preprocessing.

    Numeric ranges below reflect the min/max observed in the training data;
    they are informational only and are not enforced.
    """

    A: float = Field(description="Range seen on training: 0 - 30")
    B: float = Field(description="Range seen on training: -1 - 20")
    C: float | None = Field(
        default=None, description="Range seen on training: 0 - 617324. Could be null."
    )
    D: float = Field(description="Range seen on training: 0 - 180")
    E: float = Field(description="Range seen on training: 0 - 45")
    F: float = Field(description="Range seen on training: 0 - 1")
    G: float = Field(description="Range seen on training: 0 - 1")
    H: float = Field(description="Range seen on training: 0 - 21")
    I: float = Field(description="Range seen on training: 0 - 24")
    J: CountryCode = Field(description="Country code (ISO-like, 2 letters)")
    L: float = Field(description="Range seen on training: 0 - 7")
    M: float = Field(description="Range seen on training: 1 - 13")
    N: float = Field(description="Range seen on training: 1 - 10")
    O: float = Field(description="Range seen on training: 0 - 3")
    P: float = Field(description="Range seen on training: 1 - 41")
    S: float = Field(description="Range seen on training: -1 - 99.97")
    Monto: float = Field(description="Range seen on training: 0.05 - 998.11")
    Q: float = Field(description="Range seen on training: 0 - 984.42")
    R: float = Field(description="Range seen on training: 0 - 984.44")


class PredictionResponse(BaseModel):
    """Fraud detection prediction response containing fraud classification and probability."""

    is_fraud: bool
    fraud_probability: float
