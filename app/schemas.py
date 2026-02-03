"""Pydantic schemas for FruitDetector API."""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field, conlist, ConfigDict, AliasChoices


class FruitClass(str, Enum):
    apple = "apple"
    banana = "banana"
    tomato = "tomato"


BoundingBox = conlist(float, min_length=4, max_length=4)


class Detection(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "fruit_id": "img-123-0",
                "class": "apple",
                "confidence": 0.86,
                "bbox": [12.0, 20.0, 128.0, 256.0],
            }
        },
    )

    fruit_id: str = Field(..., description="Unique identifier for this detection.")
    class_name: FruitClass = Field(
        ...,
        alias="class",
        validation_alias="class",
        serialization_alias="class",
    )
    confidence: float = Field(..., ge=0, le=1)
    bbox: BoundingBox = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2] in pixel coordinates of original image.",
    )


class DetectionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str
    width: int
    height: int
    fruits: List[Detection] = Field(
        default_factory=list,
        alias="fruits",
        validation_alias=AliasChoices("fruits", "detections"),
        serialization_alias="fruits",
        description="All fruit detections found in the frame.",
    )

    @property
    def detections(self) -> List[Detection]:
        """Backward compatible accessor for older clients."""

        return self.fruits


class ErrorResponse(BaseModel):
    detail: str


