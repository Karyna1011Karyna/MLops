from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Текст питання користувача")


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    served_by: str = Field(..., description="Which registered model alias answered: 'champion' or 'challenger'")
    model_version: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    label: str = Field(..., description="Правильна категорія питання")


class FeedbackResponse(BaseModel):
    id: int
    status: str
