from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import base64
import io

app = FastAPI(title="PPE Safety Compliance Detector", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))

# 1. PUT THESE CONSTANTS HERE (right after loading the model, above your routes)
REQUIRED_GEARS = {"helmet", "vest", "gloves", "goggles", "boots"}

NO_TO_GEAR = {
    "no-boots": "boots",
    "no-gloves": "gloves",
    "no-goggles": "goggles",
    "no-helmet": "helmet",
    "no-vest": "vest",
}


@app.get("/")
def read_index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(image, conf=0.4)
    result = results[0]

    # 2. PUT THIS COMPLIANCE LOGIC HERE (inside @app.post("/detect"))
    detections = []
    detected_gears = set()
    explicitly_missing = set()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()

        if label in REQUIRED_GEARS:
            detected_gears.add(label)
        elif label in NO_TO_GEAR:
            explicitly_missing.add(NO_TO_GEAR[label])

        detections.append({
            "label": label,
            "confidence": round(float(box.conf[0]), 3),
            "box": [round(float(x), 1) for x in box.xyxy[0].tolist()]
        })

    unseen_gears = REQUIRED_GEARS - detected_gears
    missing_gears = sorted(list(explicitly_missing | unseen_gears))
    is_compliant = len(missing_gears) == 0 and len(explicitly_missing) == 0

    if is_compliant:
        status_message = "Compliant: All required safety gears are present."
    else:
        status_message = (
            f"Non-Compliant: Missing {', '.join(missing_gears)}. "
            "Additional safety gear is required."
        )
    
    # Keep your existing annotated_image_base64 creation lines right here
    annotated_frame=result.plot()
    annotated_rgb=Image.fromarray(annotated_frame[...,::-1])
    buffer=io.BytesIO()
    annotated_rgb.save(buffer,format="JPEG")
    annotated_image_base64=base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    
    return {
        "compliant": is_compliant,
        "status_message": status_message,
        "detected_gears": sorted(list(detected_gears)),
        "explicitly_missing": sorted(list(explicitly_missing)),
        "missing_gears": missing_gears,
        "detections": detections,
        "annotated_image_base64": annotated_image_base64,
    }