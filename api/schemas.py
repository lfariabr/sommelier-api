"""Pydantic v2 request/response models for the FastAPI service.

Request keys are snake_case for ergonomics; `to_features()` maps them back to the
dataset's spaced column names that the joblib models were trained on.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Real first row of winequality-red.csv — drives the Swagger "Try it out" example.
EXAMPLE_WINE = {
    "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red",
}


class WineFeatures(BaseModel):
    """The 12 inputs both models expect (11 physicochemical + wine_type)."""

    fixed_acidity: float = Field(..., ge=0, description="g(tartaric acid)/dm³")
    volatile_acidity: float = Field(..., ge=0, description="g(acetic acid)/dm³")
    citric_acid: float = Field(..., ge=0, description="g/dm³")
    residual_sugar: float = Field(..., ge=0, description="g/dm³")
    chlorides: float = Field(..., ge=0, description="g(sodium chloride)/dm³")
    free_sulfur_dioxide: float = Field(..., ge=0, description="mg/dm³")
    total_sulfur_dioxide: float = Field(..., ge=0, description="mg/dm³")
    density: float = Field(..., ge=0, description="g/cm³")
    pH: float = Field(..., ge=0, le=14)
    sulphates: float = Field(..., ge=0, description="g(potassium sulphate)/dm³")
    alcohol: float = Field(..., ge=0, le=100, description="% vol")
    wine_type: Literal["red", "white"] = Field("red")

    model_config = {"json_schema_extra": {"example": EXAMPLE_WINE}}

    def to_features(self) -> dict:
        return {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.pH,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol,
            "wine_type": self.wine_type,
        }


class ScoreResponse(BaseModel):
    quality: float


class GradeResponse(BaseModel):
    grade: str
    label: int
    proba_high: float
    proba_low: float


class PredictResponse(BaseModel):
    score: float
    grade: GradeResponse
