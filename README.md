### Car for Crash: Vehicle damage detection
ในกระบวนการตรวจรับรถยนต์ก่อนและหลังการใช้บริการที่ศูนย์บริการรถยนต์ ปัจจุบัน พนักงานจะทำการตรวจสอบและบันทึกข้อมูลความเสียหายของรถยนต์ลงในกระดาษ ซึ่งไม่มีหลักฐานภาพถ่ายที่ชัดเจน ส่งผลให้เกิดความผิดพลาดหรือการตีความที่ต่างกันระหว่างพนักงานกับลูกค้า โครงการนี้มีจุดประสงค์ในการออกแบบและพัฒนาระบบที่สามารถจำแนกและบันทึกความเสียหายภายนอกของรถยนต์โดยอัตโนมัติ รวมถึงการเก็บหลักฐานภาพถ่ายที่ชัดเจน เพื่อช่วยลดความผิดพลาดและทำให้กระบวนการตรวจรับรถยนต์มีความโปร่งใส

#### Tech Stack
- FastAPI
- Ultralytics YOLO
- PyTorch
- OpenCV (cv2)
- PIL (Pillow)
- ReportLab (PDF generation)
- Uvicorn

#### Requirement ในการติดตั้ง
- Python version 3.9 ขึ้นไป

#### วิธีการติดตั้ง

1) Clone repository
```
git clone https://github.com/zoneul/vehicle-damage-detection.git
```

2) สร้าง virtual environment
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3) ติดตั้ง dependencies
```
pip install -r requirements.txt
```

#### การรัน Website Server

Local development:
```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level info
```
#### โครงสร้าง dataset
```
|- train
|  |-images
|  |-labels
|- val
|  |-images
|  |-labels
|- test
|  |-images
|  |-labels
```

#### (เพิ่มเติม) การรัน Code เพื่อการทำซ้ำ
- ในการเตรียม dataset หาก dataset ที่ต้องการใช้อยู่ในรูปแบบ JSON ให้รัน Script preprocess.py 
- หากต้องการสร้าง weight model ใหม่ให้รัน Script train.py