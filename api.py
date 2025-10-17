from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import logging
from pathlib import Path
import asyncio
from typing import Optional, List
import json
import uuid
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import tempfile
import base64
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO Vehicle Damage Detection API",
    description="API สำหรับตรวจจับความเสียหายของรถยนต์ด้วย YOLO",
    version="2.0.0"
)

allowed_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://vdd.noproject-server.duckdns.org",
    "http://vdd.noproject-server.duckdns.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model variable
model = None

# Global storage for batch results
batch_results = {}

class YOLODetector:
    def __init__(self, model_path: str = "model/best.pt"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.colors = [
            (255, 0, 0),    # Red - dent
            (0, 255, 0),    # Green - scratch
            (0, 0, 255),    # Blue - crack
            (255, 255, 0),  # Yellow - shattered_glass
            (255, 0, 255),  # Magenta - broken_lamp
            (0, 255, 255),  # Cyan - flat_tire
        ]
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Get class names
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            else:
                # Fallback class names
                self.class_names = {
                    0: 'dent', 1: 'scratch', 2: 'crack', 
                    3: 'shattered_glass', 4: 'broken_lamp', 5: 'flat_tire'
                }
            
            logger.info(f"Classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5):
        """Run YOLO detection on image"""
        try:
            results = self.model(image, conf=conf_threshold)
            return results[0]  # Get first result
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise e
    
    def draw_detections(self, image: np.ndarray, results, font_scale: float = 0.6):
        """Draw bounding boxes and labels on image"""
        try:
            # Convert to PIL for better text rendering
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Try to use a better font
            try:
                font = ImageFont.truetype("arial.ttf", size=int(20 * font_scale))
            except:
                font = ImageFont.load_default()
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get class name and color
                    class_name = self.class_names.get(class_id, f"Class_{class_id}")
                    color = self.colors[class_id % len(self.colors)]
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Prepare label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get text size for background
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Draw label background
                    draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], 
                                 fill=color)
                    
                    # Draw label text
                    draw.text((x1 + 2, y1 - text_height - 3), label, 
                             fill=(255, 255, 255), font=font)
            
            # Convert back to BGR numpy array
            result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result_image
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image  # Return original image if drawing fails

# Initialize detector
try:
    detector = YOLODetector()
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detector
    if detector is None:
        try:
            detector = YOLODetector()
            logger.info("YOLO detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    """Serve test HTML page from static files"""
    return FileResponse("static/index.html")

@app.post("/detect")
async def detect_damage(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """
    Detect vehicle damage in uploaded image (single image)
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="YOLO detector not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.info(f"Processing image: {file.filename}, size: {image.shape}")
        
        # Run detection
        results = detector.detect(image, conf_threshold=conf_threshold)
        
        # Extract detection data
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                detections.append({
                    "class_id": int(class_id),
                    "class_name": detector.class_names.get(class_id, f"Class_{class_id}"),
                    "confidence": float(conf),
                    "bbox": box.tolist()
                })
        
        # Draw detections
        output_image = detector.draw_detections(image, results)
        
        # Convert to bytes
        success, buffer = cv2.imencode('.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode output image")
        
        # Return image as streaming response with detection metadata in headers
        io_buffer = io.BytesIO(buffer.tobytes())
        
        return StreamingResponse(
            io_buffer,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=detected_{file.filename}",
                "X-Detections": json.dumps(detections)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-batch")
async def detect_damage_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = 0.5
):
    """
    Detect vehicle damage in multiple uploaded images
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="YOLO detector not initialized")
    
    if len(files) > 200:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 200 images per batch")

    batch_id = str(uuid.uuid4())
    results = []
    
    try:
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/'):
                continue
            
            # Read image data
            image_data = await file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            logger.info(f"Processing batch image: {file.filename}, size: {image.shape}")
            
            # Run detection
            detection_results = detector.detect(image, conf_threshold=conf_threshold)
            
            # Extract detection data
            detections = []
            if detection_results.boxes is not None:
                boxes = detection_results.boxes.xyxy.cpu().numpy()
                confidences = detection_results.boxes.conf.cpu().numpy()
                class_ids = detection_results.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        "class_id": int(class_id),
                        "class_name": detector.class_names.get(class_id, f"Class_{class_id}"),
                        "confidence": float(conf),
                        "bbox": box.tolist()
                    })
            
            # Draw detections
            output_image = detector.draw_detections(image, detection_results)
            
            # Convert to base64 for JSON response and PDF
            success, buffer = cv2.imencode('.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                output_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            else:
                output_base64 = None
            
            # Original image to base64  
            success_orig, buffer_orig = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success_orig:
                original_base64 = base64.b64encode(buffer_orig.tobytes()).decode('utf-8')
            else:
                original_base64 = None
            
            results.append({
                "filename": file.filename,
                "detections": detections,
                "detected_image": output_base64,
                "original_image": original_base64,
                "file_size": file.size if hasattr(file, 'size') else len(image_data)
            })
        
        # Store results for PDF generation
        batch_results[batch_id] = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "conf_threshold": conf_threshold
        }
        
        return {
            "batch_id": batch_id,
            "total_images": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": detector is not None,
        "device": str(detector.device) if detector else "unknown",
        "classes": detector.class_names if detector else {}
    }
    return status

@app.post("/generate-report")
async def generate_pdf_report(request: Request):
    """Generate PDF report from detection results"""
    try:
        data = await request.json()
        results = data.get("results", [])
        conf_threshold = data.get("confidence_threshold", 0.5)
        
        if not results:
            raise HTTPException(status_code=400, detail="No results provided")
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Add custom heading style
        heading4_style = ParagraphStyle(
            'CustomHeading4',
            parent=styles['Heading4'],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=5,
        )
        styles.add(heading4_style, 'CustomHeading4')
        
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("🚗 Vehicle Damage Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary information
        total_images = len(results)
        total_detections = sum(len(r.get("detections", [])) for r in results)
        damaged_images = sum(1 for r in results if len(r.get("detections", [])) > 0)
        
        # Calculate average confidence
        all_confidences = []
        for r in results:
            for det in r.get("detections", []):
                all_confidences.append(det.get("confidence", 0))
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        summary_data = [
            ["📊 Detection Summary", ""],
            ["Total Images", str(total_images)],
            ["Total Detections Found", str(total_detections)],
            ["Images with Damage", str(damaged_images)],
            ["Average Confidence", f"{avg_confidence*100:.1f}%"],
            ["Confidence Threshold Used", f"{conf_threshold*100:.1f}%"],
            ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Damage type legend
        legend_data = [
            ["🎨 Damage Type Color Legend", ""],
            ["(Colors used in detection result images)", ""],
            ["🔴 Red Bounding Box", "Dent"],
            ["🟢 Green Bounding Box", "Scratch"],
            ["🔵 Blue Bounding Box", "Crack"],
            ["🟡 Yellow Bounding Box", "Shattered Glass"],
            ["🟣 Purple Bounding Box", "Broken Lamp"],
            ["🔵 Cyan Bounding Box", "Flat Tire"]
        ]
        
        legend_table = Table(legend_data, colWidths=[2.5*inch, 2.5*inch])
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (0, 1), colors.lightblue),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Oblique'),
            ('BACKGROUND', (0, 2), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(legend_table)
        story.append(Spacer(1, 30))
        
        # Individual image results
        story.append(Paragraph("📷 Individual Image Analysis", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        damage_names = {
            0: 'Dent', 1: 'Scratch', 2: 'Crack',
            3: 'Shattered Glass', 4: 'Broken Lamp', 5: 'Flat Tire'
        }
        
        for i, result in enumerate(results, 1):
            filename = result.get("filename", f"Image_{i}")
            detections = result.get("detections", [])
            file_size = result.get("file_size", 0)
            original_image_b64 = result.get("original_image")
            detected_image_b64 = result.get("detected_image")
            
            # Image info
            story.append(Paragraph(f"Image {i}: {filename}", styles['Heading3']))
            
            image_info = [
                ["Filename", filename],
                ["File Size", f"{file_size/1024/1024:.2f} MB" if file_size > 0 else "Unknown"],
                ["Damage Count", str(len(detections))],
                ["Status", "Damaged" if len(detections) > 0 else "No Damage Found"]
            ]
            
            info_table = Table(image_info, colWidths=[2*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT')
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 15))
            
            # Add images to PDF if available
            try:
                if original_image_b64 and detected_image_b64:
                    # Create temporary image files
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as orig_temp:
                        orig_temp.write(base64.b64decode(original_image_b64))
                        orig_temp_path = orig_temp.name
                    
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as det_temp:
                        det_temp.write(base64.b64decode(detected_image_b64))
                        det_temp_path = det_temp.name
                    
                    # Add images side by side with enhanced styling
                    image_table_data = [
                        ["📷 Original Image", "🎯 Detection Result (with Bounding Boxes)"],
                        [RLImage(orig_temp_path, width=2.4*inch, height=1.8*inch), 
                         RLImage(det_temp_path, width=2.4*inch, height=1.8*inch)]
                    ]
                    
                    image_table = Table(image_table_data, colWidths=[2.6*inch, 2.6*inch])
                    image_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('GRID', (0, 0), (-1, -1), 2, colors.black),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                        ('TOPPADDING', (0, 1), (-1, 1), 10),
                        ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
                        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
                        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.darkblue),
                    ]))
                    
                    story.append(image_table)
                    story.append(Spacer(1, 15))
                    
                    # Clean up temp files
                    Path(orig_temp_path).unlink(missing_ok=True)
                    Path(det_temp_path).unlink(missing_ok=True)
                    
            except Exception as img_error:
                logger.warning(f"Could not add images to PDF for {filename}: {img_error}")
                story.append(Paragraph("⚠️ Images could not be embedded in this report", styles['Normal']))
                story.append(Paragraph("Note: Detection results are still available in the table below", styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Detection details table
            if detections:
                detection_data = [["Damage Type", "Confidence Score", "Bounding Box (x1,y1)-(x2,y2)"]]
                for det in detections:
                    class_name = damage_names.get(det.get("class_id", -1), "Unknown")
                    confidence = det.get("confidence", 0) * 100
                    bbox = det.get("bbox", [])
                    bbox_str = f"({bbox[0]:.0f},{bbox[1]:.0f}) - ({bbox[2]:.0f},{bbox[3]:.0f})" if len(bbox) >= 4 else "N/A"
                    detection_data.append([class_name, f"{confidence:.1f}%", bbox_str])
                
                detection_table = Table(detection_data, colWidths=[1.8*inch, 1.2*inch, 2.4*inch])
                detection_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('LINEBELOW', (0, 0), (-1, 0), 2, colors.darkred),
                ]))
                
                story.append(Paragraph("🔍 Detection Details:", styles['CustomHeading4']))
                story.append(Paragraph("The colored bounding boxes in the detection result image correspond to the damage types listed below:", styles['Normal']))
                story.append(Spacer(1, 5))
                story.append(detection_table)
            else:
                story.append(Paragraph("✅ No damage detected in this image", styles['Normal']))
                story.append(Paragraph("The detection result image shows no bounding boxes as no damage was found.", styles['Normal']))
            
            story.append(Spacer(1, 25))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Return PDF as streaming response
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Vehicle_Damage_Detection_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@app.get("/info")
async def get_model_info():
    """Get model information"""
    if detector is None:
        raise HTTPException(status_code=503, detail="YOLO detector not initialized")
    
    return {
        "model_path": detector.model_path,
        "device": str(detector.device),
        "classes": detector.class_names,
        "num_classes": len(detector.class_names),
        "colors": detector.colors
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
