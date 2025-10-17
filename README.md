### README.md

#### YOLO Vehicle Damage Detection API

An end-to-end FastAPI service for vehicle damage detection using YOLO (Ultralytics). The API supports single and batch image detection, returns annotated images, and can generate a styled PDF report with summary stats and per-image analysis.

- Title: YOLO Vehicle Damage Detection API
- Description: API สำหรับตรวจจับความเสียหายของรถยนต์ด้วย YOLO
- Version: 2.0.0

#### Features

- YOLO model inference (Ultralytics) with GPU/CPU auto-selection
- Single image detection endpoint returning an annotated JPEG and JSON detections in headers
- Batch detection for up to 200 images
- PDF report generation with:
  - Summary metrics (totals, average confidence, threshold used)
  - Color-coded legend for damage types
  - Side-by-side original vs detection images
  - Detailed detection tables per image
- CORS configured for local and provided domains
- Health and model info endpoints
- Static test page served from /static

#### Tech Stack

- FastAPI
- Ultralytics YOLO
- PyTorch
- OpenCV (cv2)
- PIL (Pillow)
- ReportLab (PDF generation)
- Uvicorn

#### Endpoints

- GET /health
  - Health and model status.
  - Response:
    - status: healthy|...
    - model_loaded: bool
    - device: cpu|cuda
    - classes: class mapping

- GET /info
  - Returns model metadata (path, device, class names, colors).

- GET /
  - Serves static/index.html (test UI). If file missing, 404.

- POST /detect
  - Single image detection.
  - Form-data: file (image/*), conf_threshold (float, default 0.5)
  - Returns: image/jpeg stream (annotated)
  - Headers:
    - Content-Disposition: inline; filename=detected_<original>
    - X-Detections: JSON array of detections
      - class_id, class_name, confidence, bbox [x1, y1, x2, y2]

- POST /detect-batch
  - Multiple image detection (max 200).
  - Form-data: files (list of images), conf_threshold (float, default 0.5)
  - Returns JSON:
    - batch_id: string
    - total_images: int
    - results: array per image
      - filename
      - detections (same structure as above)
      - detected_image: base64 JPEG
      - original_image: base64 JPEG
      - file_size: bytes

- POST /generate-report
  - Generates a PDF report from results (typically from /detect-batch).
  - Body (application/json):
    - results: array (same structure returned by /detect-batch)
    - confidence_threshold: float (optional; displayed in report)
  - Returns: application/pdf (attachment)

- Static files
  - Mounted at /static; expects a static/index.html demo page.

#### Classes and Colors

Default mapping in code:
- 0: dent (Red)
- 1: scratch (Green)
- 2: crack (Blue)
- 3: shattered_glass (Yellow)
- 4: broken_lamp (Magenta)
- 5: flat_tire (Cyan)

Note: If model.model.names is available, it’s used as the authoritative class list.

#### Directory Structure

- api.py (or your main file containing the code snippet)
- model/best.pt (YOLO weights; required)
- static/
  - index.html (test UI)
- requirements.txt (see below)

#### Requirements

Create a requirements.txt similar to:

```
fastapi
uvicorn[standard]
torch
ultralytics
opencv-python
numpy
pillow
reportlab
python-multipart
```

Notes:
- torch install may vary by CUDA version. See PyTorch’s official instructions for GPU support.
- ultralytics pulls YOLOv8 code.
- On some systems you might need opencv-python-headless if deploying in minimal environments.

#### Installation

1) Clone repository and enter directory.

2) Create a virtual environment:
- Python 3.9+ recommended.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3) Install dependencies:
```
pip install -r requirements.txt
```

4) Place your YOLO weights:
- Put your trained weights at model/best.pt

5) Prepare static UI (optional):
- Ensure static/index.html exists to test via browser.

#### Running the Server

Local development:
```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

- The application auto-loads the YOLO model at startup.
- By default, runs on http://localhost:8000

CORS is preconfigured for:
- http://localhost:8000
- http://127.0.0.1:8000

Adjust allowed_origins in code as needed.

#### Usage Examples

Single image detection via curl:
```
curl -X POST "http://localhost:8000/detect?conf_threshold=0.5" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  --output detected_image.jpg -i
```
- The response headers include X-Detections with JSON metadata.

Batch detection via curl:
```
curl -X POST "http://localhost:8000/detect-batch?conf_threshold=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/img1.jpg" \
  -F "files=@/path/to/img2.jpg"
```

Generate PDF report:
- Use the results payload returned by /detect-batch as input:
```
curl -X POST "http://localhost:8000/generate-report" \
  -H "Content-Type: application/json" \
  -d @results.json \
  --output Vehicle_Damage_Detection_Report.pdf
```
Where results.json looks like:
```
{
  "confidence_threshold": 0.5,
  "results": [
    {
      "filename": "img1.jpg",
      "detections": [
        {"class_id": 0, "class_name": "dent", "confidence": 0.87, "bbox": [x1,y1,x2,y2]}
      ],
      "detected_image": "<base64-encoded-jpeg>",
      "original_image": "<base64-encoded-jpeg>",
      "file_size": 123456
    }
  ]
}
```

Health check:
```
curl http://localhost:8000/health
```

Model info:
```
curl http://localhost:8000/info
```

#### Important Implementation Notes

- Model file: The server will raise an error if model/best.pt is missing.
- Duplicate root route: The code defines GET / twice:
  - get_test_page() returning FileResponse("static/index.html")
  - read_root() reading static/index.html and returning HTMLResponse
  Keep only one to avoid FastAPI override ambiguity. Recommend removing one (prefer read_root).
- Font rendering: For label drawing, arial.ttf is attempted; if absent, defaults to PIL’s basic font. You may package a cross-platform TTF (e.g., DejaVuSans.ttf) and reference its path.
- Large batch sizes: Limited to 200 images to prevent memory pressure.
- Memory management: Temporary files for embedding images in PDF are cleaned up after use.
- Device selection: torch.cuda.is_available() automatically selects CUDA if available.

#### Error Handling

- 400 if file is not an image or cannot be decoded
- 503 if model is not initialized
- 500 for unexpected errors during detection or report generation
- /generate-report returns 400 if no results provided

#### Deployment Tips

- Use a production server like gunicorn/uvicorn workers or uvicorn directly behind a reverse proxy (Nginx).
- Set appropriate timeouts and max body size for large batch uploads.
- Ensure libGL dependencies for OpenCV are installed in Linux environments.
- Pin ultralytics and torch versions for reproducibility.

Example systemd service snippet:
```
[Unit]
Description=YOLO Vehicle Damage Detection API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/vdd
ExecStart=/opt/vdd/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

#### Security

- Validate and limit uploaded file sizes at the proxy and/or FastAPI level.
- Restrict CORS origins to trusted domains.
- Consider authentication for non-public deployments.
- Avoid exposing model internals unless required.

#### Troubleshooting

- ImportError: libGL.so not found
  - Install system packages like libgl1 (Debian/Ubuntu): apt-get update && apt-get install -y libgl1
- CUDA not used
  - Ensure compatible torch build and NVIDIA drivers/CUDA toolkit are installed.
- Model not loading
  - Verify model path and file existence; confirm ultralytics and weights compatibility.
- PDF images missing
  - Ensure base64 strings are valid JPEGs; check temporary file write permissions.

#### License

Specify your project’s license here (e.g., MIT).

#### Acknowledgements

- Ultralytics YOLO
- FastAPI
- ReportLab
- OpenCV and Pillow