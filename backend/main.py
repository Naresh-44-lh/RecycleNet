from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys, os, time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.predict import predict_image

app = FastAPI(title="Waste Classifier API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows frontend to call from any origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session stats
stats: dict = {"total": 0, "by_class": {}}


@app.get("/")
def health():
    return {"status": "ok", "model": "EfficientNetB4"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                "Use JPEG, PNG or WEBP."
            ),
        )

    # Validate file size (max 10 MB)
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    start = time.time()
    result = predict_image(image_bytes)
    result["inference_ms"] = round((time.time() - start) * 1000, 1)

    # Update session stats
    stats["total"] += 1
    c = result["class"]
    stats["by_class"][c] = stats["by_class"].get(c, 0) + 1

    return result


@app.get("/stats")
def get_stats():
    return stats
