document.addEventListener('DOMContentLoaded', () => {
  const session_id = window.location.pathname.split('/').filter(Boolean).pop();
  
  const COLORS = {
    Region: {
      MainZone: '#2563eb', MarginTextZone: '#16a34a', NumberingZone: '#ca8a04',
      DamageZone: '#dc2626', GraphicZone: '#9333ea', DigitizationArtefactZone: '#6b7280', CustomZone: '#ec4899'
    },
    Line: {
      DefaultLine: '#1f2937', HeadingLine: '#2563eb', InterlinearLine: '#16a34a'
    }
  };

  const TYPES = {
    region: Object.keys(COLORS.Region),
    baseline: Object.keys(COLORS.Line)
  };

  let canvas, currentMode = 'region';
  let isDrawing = false;
  let currentPoints = [];
  let activeObject = null;
  let isDirty = false;
  let undoStack = [];
  let autosaveTimer;

  const elements = {
    canvas: document.getElementById('main-canvas'),
    typeSelect: document.getElementById('segm-type'),
    typeLabel: document.getElementById('type-label'),
    modeBtns: document.querySelectorAll('.mode-btn'),
    saveStatus: document.getElementById('save-status'),
    transPanel: document.getElementById('trans-panel'),
    transInput: document.getElementById('trans-input'),
    stats: document.getElementById('stats')
  };

  function initCanvas() {
    canvas = new fabric.Canvas('main-canvas', {
      backgroundColor: '#eee',
      selection: false,
      stopCursorOnTClipped: true
    });

    // Zoom/Pan setup
    canvas.on('mouse:wheel', function(opt) {
      const delta = opt.e.deltaY;
      let zoom = canvas.getZoom();
      zoom *= delta > 0 ? 0.9 : 1.1;
      zoom = Math.min(Math.max(0.1, zoom), 10);
      canvas.zoomToPoint({ x: opt.e.clientX, y: opt.e.clientY }, zoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });

    canvas.on('mouse:down', function(opt) {
      const evt = this.getPointer(opt.e);
      if (opt.e.button === 1 || (opt.e.button === 0 && opt.e.spaceKey)) {
        // Pan mode
        this.isDragging = true;
        this.lastPosX = opt.e.clientX;
        this.lastPosY = opt.e.clientY;
        this.requestRenderAll();
        return;
      }
    });

    canvas.on('mouse:move', function(opt) {
      if (this.isDragging) {
        const dx = opt.e.clientX - this.lastPosX;
        const dy = opt.e.clientY - this.lastPosY;
        this.viewportTransform[4] += dx;
        this.viewportTransform[5] += dy;
        this.lastPosX = opt.e.clientX;
        this.lastPosY = opt.e.clientY;
        this.requestRenderAll();
      }
    });

    canvas.on('mouse:up', function(opt) {
      this.isDragging = false;
      canvas.requestRenderAll();
    });

    // ponytail: Fabric 5.3 Image.fromURL is callback-based (Promise form is
    // Fabric 6+). loadAutosuggest() runs first, decoupled from image load,
    // so a slow/failed image doesn't block the 26 BLLA-suggested baselines.
    loadAutosuggest();
    fabric.Image.fromURL(`/api/sessions/${session_id}/image`, (img) => {
      if (!img) {
        console.error('Failed to load fragment image');
        const banner = document.createElement('div');
        banner.textContent = 'Failed to load fragment image';
        banner.style.cssText = 'position:fixed;top:4rem;left:50%;transform:translateX(-50%);background:#dc2626;color:#fff;padding:0.75rem 1rem;border-radius:4px;z-index:1000;';
        document.body.appendChild(banner);
        return;
      }
      const wrapper = elements.canvas.parentElement;
      const availW = Math.max(200, wrapper.clientWidth - 32);
      const availH = Math.max(200, wrapper.clientHeight - 32);
      const scale = Math.min(availW / img.width, availH / img.height, 1);
      img.set({ scaleX: scale, scaleY: scale });
      canvas.setWidth(Math.round(img.width * scale));
      canvas.setHeight(Math.round(img.height * scale));
      canvas.setBackgroundImage(img, () => canvas.renderAll());
      wrapper.scrollLeft = 0;
      wrapper.scrollTop = 0;
    });

    canvas.on('mouse:down', handleMouseDown);
  }

  async function loadAutosuggest() {
    try {
      const res = await fetch(`/api/sessions/${session_id}/autosuggest`);
      const data = await res.json();

      // ponytail: Fabric 5 wants points as [{x,y},...], backend sends [[x,y],...].
      const toPts = (arr) => arr.map(p => Array.isArray(p) ? {x: p[0], y: p[1]} : p);

      (data.regions || []).forEach(r => {
        const poly = new fabric.Polygon(toPts(r.polygon), {
          fill: COLORS.Region[r.type] + '44',
          stroke: COLORS.Region[r.type],
          strokeWidth: 2,
          selectable: true,
          id: r.id,
          type: 'region',
          segmType: r.type
        });
        canvas.add(poly);
      });

      (data.lines || []).forEach(l => {
        const line = new fabric.Polyline(toPts(l.baseline), {
          stroke: COLORS.Line[l.type],
          strokeWidth: 3,
          fill: 'transparent',
          selectable: true,
          id: l.id,
          type: 'line',
          segmType: l.type,
          transcript: l.transcript || ''
        });
        canvas.add(line);
      });
      updateStats();
    } catch (err) {
      console.error('loadAutosuggest failed:', err);
      elements.stats.textContent = '· autosuggest failed — see console';
    }
  }

  function handleMouseDown(opt) {
    if (currentMode === 'transcribe') {
      if (opt.target && opt.target.type === 'line') {
        selectLine(opt.target);
      }
      return;
    }
    
    // If panning, don't draw
    if (canvas.isDragging) return;

    const pointer = canvas.getPointer(opt.e);
    const point = [pointer.x, pointer.y];

    if (!isDrawing) {
      isDrawing = true;
      currentPoints = [point];
      return;
    }

    currentPoints.push(point);
    
    if (currentPoints.length > 1) {
      const last = currentPoints[currentPoints.length - 2];
      const line = new fabric.Line([last[0], last[1], point[0], point[1]], {
        stroke: COLORS[currentMode === 'region' ? 'Region' : 'Line'][elements.typeSelect.value],
        strokeWidth: 2,
        selectable: false,
        id: 'temp'
      });
      canvas.add(line);
      setTimeout(() => canvas.remove(line), 100);
    }
  }

  function selectLine(obj) {
    activeObject = obj;
    elements.transPanel.style.display = 'flex';
    elements.transInput.value = obj.transcript || '';
    canvas.setActiveObject(obj);
    
    // Auto-advance logic: find the next line based on Y coordinate
    const lines = canvas.getObjects('polyline');
    const sortedLines = lines.sort((a, b) => a.getCenter().top - b.getCenter().top);
    const idx = sortedLines.indexOf(obj);
    if (idx < sortedLines.length - 1) {
      window.activeLineIndex = idx;
    }
  }

  function setMode(mode) {
    currentMode = mode;
    elements.modeBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    
    const group = document.getElementById('type-selector-group');
    if (mode === 'transcribe') {
      group.style.display = 'none';
      elements.transPanel.style.display = 'none';
    } else {
      group.style.display = 'flex';
      elements.typeLabel.textContent = mode === 'region' ? 'Region Type' : 'Line Type';
      elements.typeSelect.innerHTML = TYPES[mode].map(t => `<option value="${t}">${t}</option>`).join('');
    }
    
    // Color the toolbar based on mode
    const toolbar = document.querySelector('.toolbar');
    const modeColors = { region: '#2563eb', baseline: '#16a34a', transcribe: '#f59e0b' };
    toolbar.style.borderLeftColor = modeColors[mode];
    
    save(); // Autosave on mode switch
  }

  function updateStats() {
    const allRegions = canvas.getObjects('polygon');
    const allLines = canvas.getObjects('polyline');
    
    const typedRegions = allRegions.filter(r => r.segmType && r.segmType !== 'CustomZone').length;
    const typedLines = allLines.filter(l => l.transcript && l.transcript.trim() !== '').length;
    
    elements.stats.textContent = `· ${typedLines}/${allLines.length} lines transcribed, ${typedRegions}/${allRegions.length} regions typed`;
  }

  function setDirty() {
    isDirty = true;
    elements.saveStatus.textContent = 'unsaved changes';
    elements.saveStatus.classList.remove('saved');
    pushUndo();
  }

  function pushUndo() {
    const state = JSON.stringify(canvas);
    undoStack.push(state);
    if (undoStack.length > 50) undoStack.shift();
  }

  function undo() {
    if (undoStack.length === 0) return;
    const state = undoStack.pop();
    canvas.loadFromJSON(state, () => {
      canvas.renderAll();
      updateStats();
      setDirty();
    });
  }

  async function save() {
    elements.saveStatus.textContent = 'saving...';
    const regions = canvas.getObjects('polygon').map(p => ({
      id: p.id, polygon: p.getPoints().map(pt => [pt.x, pt.y]), type: p.segmType
    }));
    const lines = canvas.getObjects('polyline').map(p => ({
      id: p.id, baseline: p.getPoints().map(pt => [pt.x, pt.y]), type: p.segmType, transcript: p.transcript
    }));

    const res = await fetch(`/api/sessions/${session_id}/annotations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ regions, lines })
    });

    if (res.ok) {
      isDirty = false;
      elements.saveStatus.textContent = `saved at ${new Date().toLocaleTimeString()}`;
      elements.saveStatus.classList.add('saved');
    } else {
      elements.saveStatus.textContent = 'error';
    }
  }

  // Listeners
  elements.modeBtns.forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  document.getElementById('btn-save').addEventListener('click', save);
  document.getElementById('btn-delete').addEventListener('click', () => {
    const active = canvas.getActiveObject();
    if (active) {
      canvas.remove(active);
      setDirty();
      updateStats();
    }
  });
  document.getElementById('btn-clear').addEventListener('click', () => {
    if (confirm('SURE you want to clear all annotations?')) {
      canvas.getObjects().forEach(o => { if(o.id !== 'bg') canvas.remove(o) });
      setDirty();
      updateStats();
    }
  });

  elements.transInput.addEventListener('input', () => {
    if (activeObject) {
      activeObject.transcript = elements.transInput.value;
      setDirty();
      updateStats();
    }
  });

  elements.transInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      save();
      
      // Auto-advance
      const lines = canvas.getObjects('polyline');
      const sortedLines = lines.sort((a, b) => a.getCenter().top - b.getCenter().top);
      const idx = sortedLines.indexOf(activeObject);
      if (idx < sortedLines.length - 1) {
        selectLine(sortedLines[idx + 1]);
      }
    }
  });

  document.querySelectorAll('.palette button').forEach(btn => {
    btn.addEventListener('click', () => {
      const start = elements.transInput.selectionStart;
      const end = elements.transInput.selectionEnd;
      const val = btn.dataset.char;
      elements.transInput.setRangeText(val, start, end, 'end');
      elements.transInput.focus();
      if (activeObject) {
        activeObject.transcript = elements.transInput.value;
        setDirty();
        updateStats();
      }
    });
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    if (e.key === 'r') setMode('region');
    if (e.key === 'b') setMode('baseline');
    if (e.key === 't') setMode('transcribe');
    if (e.key === 'Delete') {
      const active = canvas.getActiveObject();
      if (active) {
        canvas.remove(active);
        setDirty();
        updateStats();
      }
    }
    if (e.key === 'Escape') {
      isDrawing = false;
      currentPoints = [];
      canvas.requestRenderAll();
    }
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      save();
    }
    if (e.ctrlKey && e.key === 'e') {
      e.preventDefault();
      window.location.href = `/api/sessions/${session_id}/export?format=page`;
    }
    if (e.ctrlKey && e.key === 'z') {
      e.preventDefault();
      undo();
    }
    
    // SegmOnto types
    if (currentMode === 'region' && e.key >= '1' && e.key <= '7') {
      const types = TYPES.region;
      const type = types[parseInt(e.key) - 1];
      if (type) {
        const active = canvas.getActiveObject();
        if (active && active.segmType) {
          active.segmType = type;
          active.stroke = COLORS.Region[type];
          active.fill = COLORS.Region[type] + '44';
          canvas.renderAll();
          setDirty();
        }
      }
    }
    if (currentMode === 'baseline' && e.key >= '1' && e.key <= '3') {
      const types = TYPES.baseline;
      const type = types[parseInt(e.key) - 1];
      if (type) {
        const active = canvas.getActiveObject();
        if (active && active.segmType) {
          active.segmType = type;
          active.stroke = COLORS.Line[type];
          canvas.renderAll();
          setDirty();
        }
      }
    }
  });

  // Auto-save timer
  setInterval(() => {
    if (isDirty) save();
  }, 30000);

  elements.canvas.addEventListener('dblclick', () => {
    if (!isDrawing) return;
    
    const type = elements.typeSelect.value;
    const color = COLORS[currentMode === 'region' ? 'Region' : 'Line'][type];
    
    let obj;
    if (currentMode === 'region') {
      obj = new fabric.Polygon(currentPoints, {
        fill: color + '44', stroke: color, strokeWidth: 2,
        type: 'region', segmType: type, selectable: true
      });
    } else {
      obj = new fabric.Polyline(currentPoints, {
        stroke: color, strokeWidth:  la.line_width || 3,
        fill: 'transparent',
        type: 'line', segmType: type, selectable: true, transcript: ''
      });
    }
    
    canvas.add(obj);
    isDrawing = false;
    currentPoints = [];
    setDirty();
    updateStats();
  });

  initCanvas();
  setMode('region');
});
