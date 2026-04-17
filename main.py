import io
import os
import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import DetrImageProcessor, DetrForObjectDetection

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "custom-model")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ────────────────────────────────────────────────
print(f"Loading model from '{MODEL_PATH}' on {DEVICE} ...")

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Build id → label map from the model config
id2label: dict[int, str] = {
    int(k): v for k, v in model.config.id2label.items()
}

print(f"Model loaded. Classes: {id2label}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bone Fracture Detection API",
    description="Upload an X-ray image and receive bounding-box predictions.",
    version="1.0.0",
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Bone Fracture Detection API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_path": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image (JPEG / PNG) and receive object detection predictions.

    Returns a list of detections, each containing:
    - label       : class name
    - class_id    : integer class id
    - confidence  : detection confidence (0-1)
    - bbox        : bounding box as [x_min, y_min, x_max, y_max] in pixels
    """
    # ── Validate content type ─────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    # ── Read & decode image ───────────────────────────────────────────────────
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    width, height = image.size

    # ── Run inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        target_sizes = torch.tensor([[height, width]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

    # ── Format response ───────────────────────────────────────────────────────
    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x_min, y_min, x_max, y_max = box.tolist()
        detections.append(
            {
                "label": id2label.get(label.item(), "unknown"),
                "class_id": label.item(),
                "confidence": round(score.item(), 4),
                "bbox": {
                    "x_min": round(x_min, 2),
                    "y_min": round(y_min, 2),
                    "x_max": round(x_max, 2),
                    "y_max": round(y_max, 2),
                },
            }
        )

    return JSONResponse(
        content={
            "image_size": {"width": width, "height": height},
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "num_detections": len(detections),
            "detections": detections,
        }
    )