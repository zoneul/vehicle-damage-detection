// /static/refactor.js
// Car wireframe hotspots UX with: confirm-on-upload, 'x' clear icon, pulsing until filled,
// and synchronization to #imageFiles for compatibility with existing script.js.

(function () {
  const ANGLES = [
    { key: 'front', label: 'หน้าตรง' },
    { key: 'back', label: 'หลังตรง' },
    { key: 'left', label: 'ซ้าย' },
    { key: 'right', label: 'ขวา' },
    { key: 'front_left_45', label: 'หน้าซ้าย 45°' },
    { key: 'front_right_45', label: 'หน้าขวา 45°' },
    { key: 'back_left_45', label: 'หลังซ้าย 45°' },
    { key: 'back_right_45', label: 'หลังขวา 45°' },
  ];

  // Confirmed files per angle
  const angleFiles = {}; // key: File

  // Temporary pending selection awaiting confirmation
  let pendingAngleKey = null;
  let pendingFile = null;

  function qs(id) { return document.getElementById(id); }
  function q(sel) { return document.querySelector(sel); }
  function qAll(sel) { return document.querySelectorAll(sel); }

  function setCoverageUI() {
    ANGLES.forEach(a => {
      const el = qs(`cov-${a.key}`);
      if (el) el.textContent = angleFiles[a.key] ? 'พร้อม' : 'ยังไม่พร้อม';
    });
  }

  function refreshHiddenFiles() {
    const hiddenMulti = qs('imageFiles');
    if (!hiddenMulti) return;

    const dt = new DataTransfer();
    ANGLES.forEach(a => {
      if (angleFiles[a.key]) dt.items.add(angleFiles[a.key]);
    });
    hiddenMulti.files = dt.files;

    // Notify legacy script
    hiddenMulti.dispatchEvent(new Event('change', { bubbles: true }));

    // Enable Detect when all 8 confirmed
    const detectBtn = qs('detectBtn');
    const allProvided = ANGLES.every(a => !!angleFiles[a.key]);
    if (detectBtn) detectBtn.disabled = !allProvided;

    setCoverageUI();
  }

  function setHotspotVisual(key, fileOrNull) {
    const hotspot = q(`.hotspot[data-angle-key="${key}"]`);
    if (!hotspot) return;

    const previewImg = hotspot.querySelector('.hotspot-preview');

    if (!fileOrNull) {
      // Clear
      hotspot.classList.remove('filled');
      hotspot.classList.add('pulse'); // resume pulsing when empty
      if (previewImg) previewImg.src = '';
      return;
    }

    // Fill
    const reader = new FileReader();
    reader.onload = ev => {
      if (previewImg) previewImg.src = ev.target.result;
      hotspot.classList.add('filled');
      hotspot.classList.remove('pulse'); // stop pulsing when filled
    };
    reader.readAsDataURL(fileOrNull);
  }

  function clearAngle(key) {
    delete angleFiles[key];
    const input = qs(`angle-${key}`);
    if (input) input.value = '';
    setHotspotVisual(key, null);
    refreshHiddenFiles();
  }

  function openConfirmDialog(key, file) {
    const dialog = qs('confirmDialog');
    const img = qs('confirmImg');
    const txt = qs('confirmText');
    const ok = qs('confirmOk');
    const cancel = qs('confirmCancel');
    if (!dialog || !img || !txt || !ok || !cancel) return;

    pendingAngleKey = key;
    pendingFile = file;

    const reader = new FileReader();
    reader.onload = ev => { img.src = ev.target.result; };
    reader.readAsDataURL(file);

    const angle = ANGLES.find(a => a.key === key);
    txt.textContent = `ภาพนี้คือมุม “${angle ? angle.label : ''}” ถูกต้องหรือไม่?`;

    ok.onclick = null;
    cancel.onclick = null;

    ok.onclick = () => {
      if (pendingAngleKey && pendingFile) {
        angleFiles[pendingAngleKey] = pendingFile;
        setHotspotVisual(pendingAngleKey, pendingFile);
        refreshHiddenFiles();
      }
      pendingAngleKey = null;
      pendingFile = null;
      dialog.close();
    };

    cancel.onclick = () => {
      const k = pendingAngleKey;
      const input = k ? qs(`angle-${k}`) : null;
      if (input) input.value = '';
      pendingAngleKey = null;
      pendingFile = null;
      dialog.close();
    };

    dialog.showModal();
  }

  function bindHotspots() {
    // Clicking hotspot opens its input; clicking the 'x' clears it.
    qAll('.hotspot').forEach(hs => {
      const key = hs.getAttribute('data-angle-key');

      // Whole hotspot opens file picker (except the clear icon)
      hs.addEventListener('click', (e) => {
        const isClear = (e.target && e.target.classList && e.target.classList.contains('hotspot-clear'));
        if (isClear) return; // handled by clear handler below
        const input = qs(`angle-${key}`);
        if (input) input.click();
      });

      // Clear icon
      const clearIcon = hs.querySelector('.hotspot-clear');
      if (clearIcon) {
        clearIcon.addEventListener('click', (e) => {
          e.stopPropagation(); // prevent opening file picker
          clearAngle(key);
        });
      }
    });
  }

  function bindAngleInputs() {
    ANGLES.forEach(a => {
      const input = qs(`angle-${a.key}`);
      if (!input) return;

      input.addEventListener('change', e => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;

        if (!file.type || !file.type.startsWith('image/')) {
          alert('ไฟล์ที่เลือกไม่ใช่รูปภาพ');
          input.value = '';
          return;
        }
        openConfirmDialog(a.key, file);
      });
    });
  }

  function bindClearAllSync() {
    const clearAllBtn = qs('clearAllBtn');
    if (clearAllBtn) {
      clearAllBtn.addEventListener('click', () => {
        ANGLES.forEach(a => clearAngle(a.key));
      });
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindHotspots();
    bindAngleInputs();
    bindClearAllSync();
    refreshHiddenFiles(); // initialize coverage + detect button
  });
})();