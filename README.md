# PPE-Safety-Compliance-Detector
A FastAPI and YOLOv8 web application that analyzes uploaded images to check workplace Personal Protective Equipment (PPE) compliance, identifies missing or required safety gear, and returns an annotated image with detection results.

## Features
- **Interactive Web Interface (`/`)**: Upload a photo in the browser to check PPE compliance, see missing gear alerts, and view the annotated output image.
- **Detection & Compliance API (`/detect`)**: Runs YOLOv8 inference using `models/best.pt`, evaluates detected and missing safety gear against required items, and returns compliance status plus a base64-encoded annotated image.
- **Health Check (`/health`)**: Returns service status (`{"status": "ok"}`).

## Model Classes
The YOLOv8 model supports 10 classes:
- **Detected gear**: `boots`, `gloves`, `goggles`, `helmet`, `vest`
- **Explicitly missing gear**: `no-boots`, `no-gloves`, `no-goggles`, `no-helmet`, `no-vest`

## Project Structure
```text
Detection Pro/
├── main.py
├── index.html
├── requirements.txt
├── detection.ipynb
└── models/
    └── best.pt
