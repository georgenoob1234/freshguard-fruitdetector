"""YOLO-based inference pipeline for the FruitDetector service."""

from __future__ import annotations

import io
import logging
from typing import List, Tuple, Optional
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from .config import Settings, get_settings
from .schemas import Detection, DetectionResponse, FruitClass

LOGGER = logging.getLogger("fruitdetector")
# Ensure the detector logger emits even when uvicorn config overrides root handlers.
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


class FruitDetector:
    """Thin wrapper around a YOLO model that enforces project constraints."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.model = self._load_model(self.settings.model_path)
        self.class_names = self._extract_class_names()
        LOGGER.info("FruitDetector initialized with confidence_threshold=%.4f", self.settings.confidence_threshold)

    def _load_model(self, weights_path: str) -> YOLO:
        try:
            return YOLO(weights_path)
        except Exception as exc:  # pragma: no cover - startup failure should crash fast
            LOGGER.exception("Failed to load YOLO model from %s", weights_path)
            raise RuntimeError(f"Unable to load YOLO model: {exc}") from exc

    def _extract_class_names(self) -> dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(idx): name.lower() for idx, name in names.items()}
        return {idx: name.lower() for idx, name in enumerate(names)}

    def _read_image(self, image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                rgb_image = img.convert("RGB")
                width, height = rgb_image.size
                # Ultralytics expects numpy arrays in BGR order.
                bgr_image = np.asarray(rgb_image)[:, :, ::-1].copy()
                return bgr_image, (width, height)
        except UnidentifiedImageError as exc:
            LOGGER.warning("Invalid image upload: %s", exc)
            raise ValueError("Invalid image file") from exc

    def detect(self, image_bytes: bytes, image_id: Optional[str] = None, imgsz: Optional[int] = None) -> DetectionResponse:
        np_image, (width, height) = self._read_image(image_bytes)
        inference_size = imgsz or self.settings.default_imgsz
        response_id = image_id or str(uuid4())

        try:
            results = self.model.predict(
                source=np_image,
                imgsz=inference_size,
                conf=self.settings.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("Inference failed: %s", exc)
            raise RuntimeError("Inference failure") from exc

        detections = self._build_detections(results, response_id, width, height)
        return DetectionResponse(
            image_id=response_id,
            width=width,
            height=height,
            fruits=detections,
        )

    def _build_detections(self, results, image_id: str, width: int, height: int) -> List[Detection]:
        if not results:
            return []

        first = results[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((0,))
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((0,), dtype=int)

        orig_h, orig_w = first.orig_shape
        scale_x = width / orig_w if orig_w else 1.0
        scale_y = height / orig_h if orig_h else 1.0

        detections: List[Detection] = []
        threshold = float(self.settings.confidence_threshold)

        for idx, (bbox, conf_val, cls_id) in enumerate(zip(xyxy, confs, classes)):
            conf = float(conf_val)
            if not np.isfinite(conf) or conf < threshold:
                LOGGER.info(
                    "Filtered out detection idx=%d with confidence=%.4f (threshold=%.4f)",
                    idx,
                    conf,
                    threshold,
                )
                continue

            class_label = self.class_names.get(int(cls_id), "")
            if class_label not in self.settings.allowed_classes:
                LOGGER.debug("Skipping class '%s' (not in allowed set)", class_label)
                continue

            scaled_bbox = [
                max(0.0, min(width, float(bbox[0] * scale_x))),
                max(0.0, min(height, float(bbox[1] * scale_y))),
                max(0.0, min(width, float(bbox[2] * scale_x))),
                max(0.0, min(height, float(bbox[3] * scale_y))),
            ]

            try:
                fruit_class = FruitClass(class_label)
            except ValueError:
                LOGGER.debug("Unknown class label '%s' skipped", class_label)
                continue

            detections.append(
                Detection(
                    fruit_id=f"{image_id}-{idx}",
                    class_name=fruit_class,
                    confidence=float(conf),
                    bbox=scaled_bbox,
                )
            )

        return detections



