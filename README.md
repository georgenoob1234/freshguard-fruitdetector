# FruitDetector Service

Passive FastAPI microservice that exposes a single `/detect` endpoint for the Brain
to request fruit detections. The implementation follows the constraints in
`MasterPrompt.txt`.

## Features

- Deterministic YOLO inference (single weights file, arbitrary `imgsz`)
- Strict API schema with Pydantic validation
- Structured error handling and lightweight logging

## Setup

```bash
cd fruitdetector
/opt/anaconda/bin/pip install -r requirements.txt
```

Place the trained YOLO weights at `models/fruitdetector.pt` or point the
`FRUIT_MODEL_PATH` environment variable to your model file.

## Running

```bash
/opt/anaconda/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

`POST /detect`

- Multipart form field `file`: RGB image (binary)
- Optional query `imgsz`: inference size (defaults to 320; fallback 416 supported)

Response body:

```json
{
  "image_id": "string",
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "fruit_id": "image-0",
      "class": "apple",
      "confidence": 0.92,
      "bbox": [12, 24, 300, 420]
    }
  ]
}
```

## Example Request

```bash
curl -X POST "http://localhost:8000/detect?imgsz=320" \
  -F "file=@/path/to/image.jpg"
```

## Health Check

- `GET /health` → `{"status": "ok"}`

---

## Docker

### Building the Image

```bash
docker build -t fruitdetector:latest .
```

### Running the Container

```bash
docker run --rm -p 8300:8300 \
  -e SERVICE_PORT=8300 \
  fruitdetector:latest
```

#### With Custom Configuration

```bash
docker run --rm -p 8300:8300 \
  -e SERVICE_PORT=8300 \
  -e FRUIT_CONFIDENCE_THRESHOLD=0.25 \
  -e FRUIT_DEFAULT_IMGSZ=416 \
  fruitdetector:latest
```

### Using Docker Compose

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# Stop
docker compose down
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_PORT` | `8300` | Port the service listens on |
| `FRUIT_MODEL_PATH` | `models/fruitdetector.pt` | Path to YOLO weights file |
| `FRUIT_CONFIDENCE_THRESHOLD` | `0.1` | Minimum confidence for detections |
| `FRUIT_DEFAULT_IMGSZ` | `320` | Default inference resolution |
| `FRUIT_FALLBACK_IMGSZ` | `416` | Fallback inference resolution |

### Health Check

The container includes a health check that polls `/health` every 30 seconds:

```bash
curl http://localhost:8300/health
# {"status": "ok"}
```






