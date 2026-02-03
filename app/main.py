"""FastAPI entrypoint for the FruitDetector microservice."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status

from .config import get_settings
from .detector import FruitDetector
from .schemas import DetectionResponse, ErrorResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("fruitdetector.api")
# Prevent duplicate logs while still emitting via a dedicated handler if needed.
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

app = FastAPI(
    title="FruitDetector Service",
    description="Passive detection service that responds to Brain requests.",
    version="1.0.0",
)


@app.on_event("startup")
def _warmup_detector() -> None:
    """Instantiate the detector once at startup so logs show immediately."""
    detector = resolve_detector()
    LOGGER.info(
        "Detector warm-up complete with confidence_threshold=%.4f",
        detector.settings.confidence_threshold,
    )


@lru_cache
def _detector_singleton() -> FruitDetector:
    return FruitDetector(settings=get_settings())


def resolve_detector() -> FruitDetector:
    return _detector_singleton()


@app.post(
    "/detect-fruits",
    response_model=DetectionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def detect(
    file: UploadFile = File(..., description="RGB image provided as multipart/form-data."),
    imgsz: Annotated[
        int | None,
        Query(
            ge=64,
            le=2048,
            description="Optional inference resolution; defaults to 320 with a 416 fallback.",
        ),
    ] = None,
    detector: FruitDetector = Depends(resolve_detector),
) -> DetectionResponse:
    raw_image = await file.read()
    if not raw_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    image_id = Path(file.filename).stem or str(uuid4())

    try:
        return detector.detect(image_bytes=raw_image, image_id=image_id, imgsz=imgsz)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Unhandled runtime error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@app.get("/health", response_model=dict[str, str])
def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["app"]


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


