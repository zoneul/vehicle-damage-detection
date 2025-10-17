// Global variables
let selectedFiles = [];
let detectionResults = [];

// Color mapping for different damage types
const damageColors = {
    0: 'rgb(255,0,0)',     // dent
    1: 'rgb(0,255,0)',     // scratch
    2: 'rgb(0,0,255)',     // crack
    3: 'rgb(255,255,0)',   // shattered_glass
    4: 'rgb(255,0,255)',   // broken_lamp
    5: 'rgb(0,255,255)'   // flat_tire
};

const damageNames = {
    0: 'รอยบุบ',
    1: 'รอยขีดข่วน',
    2: 'รอยแตก',
    3: 'กระจกแตก',
    4: 'ไฟเสีย',
    5: 'ยางแบน'
};

// DOM elements
const imageFiles = document.getElementById('imageFiles');
const detectBtn = document.getElementById('detectBtn');
const downloadPdfBtn = document.getElementById('downloadPdfBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const selectedFilesDiv = document.getElementById('selectedFiles');
const filesList = document.getElementById('filesList');
const loadingText = document.getElementById('loadingText');
const progressFill = document.getElementById('progressFill');

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    imageFiles.addEventListener('change', function () {
        selectedFiles = Array.from(this.files);
        updateFilesList();
        detectBtn.disabled = selectedFiles.length === 0;
        downloadPdfBtn.disabled = true;
        downloadZipBtn.disabled = true;
    });
});

function updateFilesList() {
    if (selectedFiles.length === 0) {
        selectedFilesDiv.style.display = 'none';
        return;
    }

    selectedFilesDiv.style.display = 'block';
    filesList.innerHTML = '';

    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <span>📁 ${file.name}</span>
                <span class="file-size">(${(file.size / 1024 / 1024).toFixed(2)} MB)</span>
            </div>
            <button class="remove-file" onclick="removeFile(${index})">❌</button>
        `;
        filesList.appendChild(fileItem);
    });

    const dt = new DataTransfer();
    selectedFiles.forEach(file => dt.items.add(file));
    imageFiles.files = dt.files;
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFilesList();
    detectBtn.disabled = selectedFiles.length === 0;
    downloadPdfBtn.disabled = true;
    downloadZipBtn.disabled = true;

    const dt = new DataTransfer();
    selectedFiles.forEach(file => dt.items.add(file));
    imageFiles.files = dt.files;
}

function clearAllFiles() {
    selectedFiles = [];
    updateFilesList();
    detectBtn.disabled = true;
    downloadPdfBtn.disabled = true;
    downloadZipBtn.disabled = true;

    imageFiles.value = '';

    results.style.display = 'none';
    detectionResults = [];
}

async function detectDamage() {
    if (selectedFiles.length === 0) {
        alert('กรุณาเลือกไฟล์รูปภาพก่อน');
        return;
    }

    loading.style.display = 'block';
    results.style.display = 'none';
    detectBtn.disabled = true;
    downloadPdfBtn.disabled = true;
    downloadZipBtn.disabled = true;
    detectionResults = [];

    try {
        loadingText.textContent = `กำลังประมวลผล ${selectedFiles.length} ไฟล์...`;
        progressFill.style.width = '50%';

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch('https://vdd.noproject-server.duckdns.org/detect-batch', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const batchResult = await response.json();
        progressFill.style.width = '90%';

        detectionResults = batchResult.results.map(result => ({
            filename: result.filename,
            originalUrl: `data:image/jpeg;base64,${result.original_image}`,
            detectedUrl: `data:image/jpeg;base64,${result.detected_image}`,
            detections: result.detections,
            fileSize: result.file_size
        }));

        displayResults();
        downloadPdfBtn.disabled = false;
        downloadZipBtn.disabled = false;

    } catch (error) {
        console.error('Error:', error);
        alert('เกิดข้อผิดพลาดในการตรวจจับ: ' + error.message);
    } finally {
        loading.style.display = 'none';
        detectBtn.disabled = false;
        progressFill.style.width = '100%';
    }
}

function initImageOverlay() {
    if (document.getElementById('imageOverlay')) return;

    const overlay = document.createElement('div');
    overlay.id = 'imageOverlay';
    overlay.style.cssText = `
        position: fixed; inset: 0; background: rgba(0,0,0,0.85);
        display: none; z-index: 9999; padding: 24px; box-sizing: border-box;
    `;

    overlay.innerHTML = `
        <div id="overlayContent" style="
            max-width: 96vw; max-height: 92vh; margin: 0 auto; display: flex; flex-direction: column; gap: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; color:#fff;">
                <div id="overlayTitle" style="font-weight:600; font-size:16px;"></div>
                <button id="overlayCloseBtn" aria-label="Close" style="
                    background: transparent; border: 1px solid #fff3; color:#fff; border-radius:8px;
                    padding:6px 10px; cursor:pointer;">✕ ปิด</button>
            </div>
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                width: 100%;
                height: calc(100vh - 60px); /* Full viewport minus any header */
                box-sizing: border-box;
                padding: 10px;
            ">
                <!-- Original Image Panel -->
                <div style="
                    background: #111;
                    border-radius: 8px;
                    padding: 10px;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                ">
                    <div style="color: #fff; font-size: 13px; margin-bottom: 6px;">📷 รูปต้นฉบับ</div>
                    <div style="
                        flex: 1;
                        overflow: auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <img id="overlayOriginal" alt="Overlay Original"
                            style="width: 100%; height: 100%; object-fit: contain;" />
                    </div>
                </div>

                <!-- Detected Image Panel -->
                <div style="
                    background: #111;
                    border-radius: 8px;
                    padding: 10px;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                ">
                    <div style="color: #fff; font-size: 13px; margin-bottom: 6px;">🎯 ผลการตรวจจับ</div>
                    <div style="
                        flex: 1;
                        overflow: auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <img id="overlayDetected" alt="Overlay Detected"
                            style="width: 100%; height: 100%; object-fit: contain;" />
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    const close = () => overlay.style.display = 'none';
    document.getElementById('overlayCloseBtn').addEventListener('click', close);
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) close();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.style.display === 'block') close();
    });
}

function openImageOverlay({ originalUrl, detectedUrl, title }) {
    const overlay = document.getElementById('imageOverlay');
    if (!overlay) return;

    document.getElementById('overlayOriginal').src = originalUrl || '';
    document.getElementById('overlayDetected').src = detectedUrl || '';
    document.getElementById('overlayTitle').textContent = title || '';
    overlay.style.display = 'block';
}

function displayResults() {
    initImageOverlay();

    const imagesGrid = document.getElementById('imagesGrid');
    const resultsEl = results;
    imagesGrid.innerHTML = '';

    let totalDetections = 0;
    let damagedImages = 0;
    let totalConfidence = 0;
    let confidenceCount = 0;

    if (!Array.isArray(detectionResults) || detectionResults.length === 0) {
        imagesGrid.innerHTML = `
            <div class="image-card" style="text-align:center; padding:20px;">
                <h4>🗂️ ยังไม่มีรูปภาพให้แสดง</h4>
                <p style="color:#6c757d;">อัปโหลดรูปภาพเพื่อเริ่มการตรวจจับความเสียหาย</p>
            </div>
        `;
        document.getElementById('totalImages').textContent = 0;
        document.getElementById('totalDetections').textContent = 0;
        document.getElementById('damagedImages').textContent = 0;
        document.getElementById('averageConfidence').textContent = '0%';
        resultsEl.style.display = 'block';
        return;
    }

    const frag = document.createDocumentFragment();

    detectionResults.forEach(result => {
        const hasDetections = result.detections && result.detections.length > 0;
        if (hasDetections) {
            damagedImages++;
            totalDetections += result.detections.length;
            result.detections.forEach(det => {
                totalConfidence += det.confidence || 0;
                confidenceCount++;
            });
        }

        const imageCard = document.createElement('div');
        imageCard.className = 'image-card';

        let detectionsHtml = hasDetections ? `
            <div class="detections-list">
                <h4>🎯 ความเสียหายที่พบ 
                    <span class="detection-count" style="background:#ffeeba; color:#856404; padding:2px 8px; border-radius:12px; font-size:12px; margin-left:6px;">
                        ${result.detections.length} รายการ
                    </span>
                </h4>
                ${result.detections.map(det => {
            const classId = det.class_id;
            const color = (damageColors && damageColors[classId]) || '#007bff';
            const name = (damageNames && damageNames[classId]) || 'ไม่ทราบประเภท';
            const conf = ((det.confidence || 0) * 100).toFixed(1);
            return `
                        <div class="detection-item" style="border-left-color:${color}">
                            <span class="detection-name">${name}</span>
                            <span class="detection-confidence" title="ความเชื่อมั่น">${conf}%</span>
                        </div>
                    `;
        }).join('')}
            </div>
        ` : `
            <div class="detections-list">
                <h4 style="color:#28a745; display:flex; align-items:center; gap:6px;">
                    ✅ ไม่พบความเสียหาย
                    <span class="detection-count" style="background:#e9f7ef; color:#28a745; padding:2px 8px; border-radius:12px; font-size:12px;">
                        0
                    </span>
                </h4>
            </div>
        `;

        const safeFilename = result.filename || 'ภาพไม่มีชื่อ';
        const originalAlt = `Original ${safeFilename}`;
        const detectedAlt = `Detected ${safeFilename}`;

        imageCard.innerHTML = `
            <div class="image-header" style="display:flex; justify-content:space-between; align-items:center; overflow-x: auto;">
                <span class="image-title" title="${safeFilename}">📁 ${safeFilename}</span>
                <span class="detection-count" style="background:#f1f3f5; color:#495057; padding:2px 8px; border-radius:12px; font-size:12px;">
                    ${result.detections?.length || 0} ความเสียหาย
                </span>
            </div>
            <div class="image-comparison">
                <div class="image-section">
                    <h4>📷 รูปต้นฉบับ</h4>
                    <img class="click-inspect" data-role="original" 
                         src="${result.originalUrl}" alt="${originalAlt}" loading="lazy"
                         style="cursor:zoom-in;">
                </div>
                <div class="image-section">
                    <h4>🎯 ผลการตรวจจับ</h4>
                    <img class="click-inspect" data-role="detected" 
                         src="${result.detectedUrl}" alt="${detectedAlt}" loading="lazy"
                         style="cursor:zoom-in;">
                </div>
            </div>
            ${detectionsHtml}
        `;

        imageCard.querySelectorAll('img.click-inspect').forEach(img => {
            img.addEventListener('click', () => {
                openImageOverlay({
                    originalUrl: result.originalUrl,
                    detectedUrl: result.detectedUrl,
                    title: `🔎 ตรวจสอบ: ${safeFilename}`
                });
            });
        });

        frag.appendChild(imageCard);
    });

    imagesGrid.appendChild(frag);

    document.getElementById('totalImages').textContent = detectionResults.length;
    document.getElementById('totalDetections').textContent = totalDetections;
    document.getElementById('damagedImages').textContent = damagedImages;
    document.getElementById('averageConfidence').textContent =
        confidenceCount > 0 ? `${(totalConfidence / confidenceCount * 100).toFixed(1)}%` : '0%';

    resultsEl.style.display = 'block';
}

async function downloadPDF() {
    if (detectionResults.length === 0) {
        alert('ไม่มีผลการตรวจจับสำหรับสร้างรายงาน');
        return;
    }

    try {
        downloadPdfBtn.disabled = true;
        loadingText.textContent = 'กำลังสร้างรายงาน PDF...';
        loading.style.display = 'block';

        const response = await fetch('/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                results: detectionResults.map(r => ({
                    filename: r.filename,
                    detections: r.detections,
                    fileSize: r.fileSize
                }))
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Vehicle_Damage_Detection_Report_${new Date().toISOString().slice(0, 10)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error:', error);
        alert('เกิดข้อผิดพลาดในการสร้างรายงาน: ' + error.message);
    } finally {
        loading.style.display = 'none';
        downloadPdfBtn.disabled = false;
    }
}

async function downloadImagesZip() {
    if (detectionResults.length === 0) {
        alert('ไม่มีรูปภาพสำหรับดาวน์โหลด');
        return;
    }

    try {
        downloadZipBtn.disabled = true;
        loadingText.textContent = 'กำลังสร้างไฟล์ ZIP...';
        loading.style.display = 'block';

        const JSZip = window.JSZip || await loadJSZip();
        const zip = new JSZip();

        const originalFolder = zip.folder("original_images");
        const detectedFolder = zip.folder("detected_images");

        for (let i = 0; i < detectionResults.length; i++) {
            const result = detectionResults[i];
            const filename = result.filename;
            const baseFilename = filename.split('.')[0];
            const extension = filename.split('.').pop();

            const originalBase64 = result.originalUrl.split(',')[1];
            const originalBlob = base64ToBlob(originalBase64, 'image/jpeg');
            originalFolder.file(`${baseFilename}_original.${extension}`, originalBlob);

            const detectedBase64 = result.detectedUrl.split(',')[1];
            const detectedBlob = base64ToBlob(detectedBase64, 'image/jpeg');
            detectedFolder.file(`${baseFilename}_detected.${extension}`, detectedBlob);

            const progress = ((i + 1) / detectionResults.length * 90);
            progressFill.style.width = `${progress}%`;
        }

        loadingText.textContent = 'กำลังบีบอัดไฟล์...';
        const zipBlob = await zip.generateAsync({ type: "blob" });

        const url = URL.createObjectURL(zipBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Vehicle_Damage_Detection_Images_${new Date().toISOString().slice(0, 10)}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error creating ZIP:', error);
        alert('เกิดข้อผิดพลาดในการสร้างไฟล์ ZIP: ' + error.message);
    } finally {
        loading.style.display = 'none';
        downloadZipBtn.disabled = false;
        progressFill.style.width = '100%';
    }
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

async function loadJSZip() {
    return new Promise((resolve, reject) => {
        if (window.JSZip) {
            resolve(window.JSZip);
            return;
        }

        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
        script.onload = () => resolve(window.JSZip);
        script.onerror = () => reject(new Error('Failed to load JSZip'));
        document.head.appendChild(script);
    });
}