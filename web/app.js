// PicoClaw 前端 App.js
let currentMusicId  = null;
let currentChoreoId = null;   // 当前选中的编排 ID
let wsConnection    = null;
let _taskPollTimer  = null;
let _choreoTaskId   = null;
let _activeTheme    = '';

// ── 初始化 ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    connectWebSocket();
    checkSystemStatus();
    loadMusicLibrary();
    loadThemeCatalog();
    loadFxCatalog();
    setInterval(checkSystemStatus, 5000);

    const audio = document.getElementById('audioPlayer');
    audio.addEventListener('timeupdate', () => {
        document.getElementById('audioTime').textContent =
            audio.currentTime.toFixed(1) + 's';
    });
    setInterval(() => {
        if (!audio.paused && wsConnection &&
            wsConnection.readyState === WebSocket.OPEN) {
            wsConnection.send(JSON.stringify({
                type: 'audio_clock', t: audio.currentTime
            }));
        }
    }, 200);

    const slider = document.getElementById('themeStrength');
    slider.addEventListener('input', () => {
        document.getElementById('themeStrengthVal').textContent =
            slider.value + '%';
    });

    // 点击 modal 背景关闭
    document.getElementById('choreoModal').addEventListener('click', e => {
        if (e.target === e.currentTarget) closeChoreoDialog();
    });
});

function setupEventListeners() {
    document.getElementById('musicFile').addEventListener('change', handleFileSelect);
}

// ── 系统状态 ────────────────────────────────────────────────────────
async function checkSystemStatus() {
    try {
        const data = await fetchJSON('/api/health');
        const sys = document.getElementById('sysStatus');
        sys.textContent = '在线';
        sys.className = 'badge ok';
        setBadge('bpuStatus',    data.bpu === 'ok'    ? '正常' : data.bpu,    data.bpu === 'ok');
        setBadge('cameraStatus', data.camera === 'ok' ? '就绪' : data.camera, data.camera === 'ok');
        setBadge('fxStatus',     (data.fx_count || 0) + ' FX', true);
    } catch {
        const s = document.getElementById('sysStatus');
        s.textContent = '离线';
        s.className = 'badge error';
    }
}

function setBadge(id, text, ok) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className   = 'badge ' + (ok ? 'ok' : 'warning');
}

// ── 视频流错误处理 ──────────────────────────────────────────────────
function handleStreamError(img) {
    setTimeout(() => { img.src = '/api/video/stream?' + Date.now(); }, 1000);
}

// ── 音乐库 ──────────────────────────────────────────────────────────
async function loadMusicLibrary() {
    const hint = document.getElementById('libraryHint');
    try {
        const data = await fetchJSON('/api/music/library');
        const sel  = document.getElementById('musicLibrarySelect');
        while (sel.options.length > 1) sel.remove(1);
        if (!data.items || data.items.length === 0) {
            hint.textContent = '库中暂无音乐';
            return;
        }
        for (const item of data.items) {
            const bpm   = item.tempo ? ` ${Math.round(item.tempo)} BPM` : '';
            const dur   = item.duration ? ` ${item.duration.toFixed(0)}s` : '';
            const flag  = item.has_choreo ? ' ✓' : '';
            const label = `${item.filename}${bpm}${dur}${flag}`;
            const opt   = new Option(label, item.music_id);
            opt.dataset.hasChoreo = item.has_choreo ? '1' : '';
            opt.dataset.status    = item.status || '';
            sel.add(opt);
        }
        hint.textContent = `共 ${data.items.length} 首（✓ 已有编排）`;
    } catch (e) {
        hint.textContent = '加载失败';
        console.warn('loadMusicLibrary:', e);
    }
}

async function onLibrarySelect(musicId) {
    if (!musicId) return;
    currentMusicId  = musicId;
    currentChoreoId = null;

    document.getElementById('choreoSummary').style.display = 'none';
    document.getElementById('taskSection').style.display   = 'none';
    document.getElementById('uploadProgress').classList.add('hidden');

    try {
        const info = await fetchJSON(`/api/music/${musicId}`);
        document.getElementById('fileInfo').classList.remove('hidden');
        document.getElementById('fileName').textContent   = info.filename || musicId;
        document.getElementById('fileDuration').textContent =
            info.duration ? info.duration.toFixed(1) + 's' : '-';
        document.getElementById('beatInfo').textContent   =
            info.tempo ? info.tempo.toFixed(0) + ' BPM' : '-';

        const analyzed = info.status === 'analyzed';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('choreBtn').disabled   = !analyzed;
        document.getElementById('previewBtn').disabled = true;

        if (analyzed) {
            await loadChoreoList(musicId);
        }
    } catch (e) {
        showTask('❌ 加载音乐信息失败: ' + e.message);
    }
}

// ── 编排列表 ────────────────────────────────────────────────────────
async function loadChoreoList(musicId) {
    try {
        const data = await fetchJSON(`/api/music/${musicId}/choreos`);
        const choreos = data.choreos || [];
        renderChoreoList(choreos);
        if (choreos.length > 0) {
            // 默认选中最后一套（最新）
            const latest = choreos[choreos.length - 1];
            setActiveChoreo(latest.choreo_id, latest.theme);
        }
    } catch (e) {
        console.warn('loadChoreoList:', e);
    }
}

function renderChoreoList(choreos) {
    const sec  = document.getElementById('choreoSummary');
    const list = document.getElementById('choreoList');
    list.innerHTML = '';

    if (choreos.length === 0) {
        sec.style.display = 'none';
        return;
    }
    sec.style.display = 'block';

    choreos.forEach((c, idx) => {
        const item = document.createElement('div');
        item.className = 'choreo-item';
        item.dataset.choreoId = c.choreo_id;

        const srcClass = c.source === 'rule' ? 'choreo-src rule' : 'choreo-src';
        const srcLabel = c.source === 'rule' ? '规则' : 'AI';
        const styleText = c.style || 'energetic';
        const themeText = c.theme ? c.theme.replace('theme_', '') : '';
        const themeSpan = themeText ? `<span>🎨 ${themeText}</span>` : '';

        item.innerHTML = `
            <div class="choreo-item-head">
                <span class="choreo-idx">#${idx + 1}</span>
                <span class="${srcClass}">${srcLabel}</span>
                <span class="choreo-style-text">${escHtml(styleText)}</span>
            </div>
            <div class="choreo-item-meta">
                ${themeSpan}
                <span>📋 ${c.track_count} 事件</span>
            </div>`;

        item.addEventListener('click', () => setActiveChoreo(c.choreo_id, c.theme));
        list.appendChild(item);
    });
}

function setActiveChoreo(choreoId, theme) {
    currentChoreoId = choreoId;

    // 更新列表高亮
    document.querySelectorAll('.choreo-item').forEach(el => {
        el.classList.toggle('active', el.dataset.choreoId === choreoId);
    });

    // 解锁预览
    document.getElementById('previewBtn').disabled = false;

    // 同步主题下拉
    if (theme) {
        document.getElementById('themeSelect').value = theme;
        _activeTheme = theme;
        updateThemeBadge(theme);
    }
}

function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── 主题滤镜 ────────────────────────────────────────────────────────
async function loadThemeCatalog() {
    try {
        const data = await fetchJSON('/api/fx/themes');
        const sel  = document.getElementById('themeSelect');
        for (const t of (data.themes || [])) {
            const label = t.description ? `${t.id}  —  ${t.description}` : t.id;
            sel.add(new Option(label, t.id));
        }
    } catch (e) {
        console.warn('loadThemeCatalog:', e);
    }
}

async function applyTheme(themeId) {
    const strength = parseInt(document.getElementById('themeStrength').value, 10) / 100;
    document.getElementById('themeStrengthVal').textContent = Math.round(strength * 100) + '%';
    try {
        await fetchJSON('/api/session/theme', 'POST', { theme_id: themeId || '', strength });
        _activeTheme = themeId;
        updateThemeBadge(themeId);
    } catch (e) {
        console.warn('applyTheme:', e);
    }
}

function updateThemeBadge(themeId) {
    const el = document.getElementById('themeActive');
    if (themeId) {
        el.classList.remove('hidden');
        el.textContent = '主题: ' + themeId.replace('theme_', '');
    } else {
        el.classList.add('hidden');
    }
}

// ── 文件选择 / 上传 ─────────────────────────────────────────────────
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileDuration').textContent = '计算中...';
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const reader = new FileReader();
        reader.onload = ev => ctx.decodeAudioData(ev.target.result, buf => {
            const m = Math.floor(buf.duration / 60);
            const s = Math.floor(buf.duration % 60);
            document.getElementById('fileDuration').textContent =
                `${m}:${String(s).padStart(2,'0')}`;
        });
        reader.readAsArrayBuffer(file);
    } catch {}
    uploadMusicFile(file);
}

async function uploadMusicFile(file) {
    const prog = document.getElementById('uploadProgress');
    prog.classList.remove('hidden');
    document.getElementById('progressFill').style.width = '0%';

    return new Promise(resolve => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', e => {
            if (e.lengthComputable) {
                const pct = Math.round(e.loaded / e.total * 100);
                document.getElementById('progressFill').style.width = pct + '%';
                document.getElementById('progressText').textContent = `上传中: ${pct}%`;
            }
        });
        xhr.addEventListener('load', () => {
            try {
                if (xhr.status === 200) {
                    const resp = JSON.parse(xhr.responseText);
                    currentMusicId = resp.music_id;
                    document.getElementById('progressText').textContent = '✅ 上传完成';
                    document.getElementById('analyzeBtn').disabled = false;
                    loadMusicLibrary();
                } else {
                    document.getElementById('progressText').textContent =
                        `❌ 上传失败 (HTTP ${xhr.status})`;
                }
            } catch (e) {
                document.getElementById('progressText').textContent = '❌ 响应解析失败: ' + e.message;
            }
            resolve();
        });
        xhr.addEventListener('error',   () => { document.getElementById('progressText').textContent = '❌ 网络错误'; resolve(); });
        xhr.addEventListener('timeout', () => { document.getElementById('progressText').textContent = '❌ 上传超时'; resolve(); });
        xhr.timeout = 120000;
        xhr.open('PUT', '/api/music/upload');
        xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');
        xhr.send(file);
    });
}

// ── 分析节奏 ────────────────────────────────────────────────────────
async function analyzMusic() {
    if (!currentMusicId) return;
    document.getElementById('analyzeBtn').disabled = true;
    showTask('节奏分析中...');
    try {
        const resp = await fetchJSON(`/api/music/${currentMusicId}/analyze`, 'POST');
        await pollTask(resp.task_id, '节奏分析');

        const info = await fetchJSON(`/api/music/${currentMusicId}`);
        showTask(`✅ 分析完成  BPM: ${info.tempo ? info.tempo.toFixed(1) : '?'}  时长: ${info.duration ? info.duration.toFixed(1)+'s' : '?'}  节拍: ${info.beat_count}`);
        document.getElementById('beatInfo').textContent = info.tempo ? info.tempo.toFixed(0) + ' BPM' : '-';
        document.getElementById('choreBtn').disabled = false;
        loadMusicLibrary();
    } catch (err) {
        showTask('❌ 分析失败: ' + err.message);
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// ── 编排对话框 ──────────────────────────────────────────────────────
function openChoreoDialog() {
    document.getElementById('styleInput').value = '';
    document.getElementById('choreoModal').classList.remove('hidden');
    setTimeout(() => document.getElementById('styleInput').focus(), 80);
}

function closeChoreoDialog() {
    document.getElementById('choreoModal').classList.add('hidden');
}

function setStyleTag(text) {
    document.getElementById('styleInput').value = text;
    document.getElementById('styleInput').focus();
}

async function confirmChoreo() {
    const style = document.getElementById('styleInput').value.trim() || 'energetic';
    closeChoreoDialog();
    await _doGenerateChoreo(style);
}

// ── 生成编排（主按钮 → 打开对话框）────────────────────────────────
function generateChoreo() {
    if (!currentMusicId) return;
    openChoreoDialog();
}

async function _doGenerateChoreo(style) {
    document.getElementById('choreBtn').disabled = true;
    showTask(`AI 编排生成中 (MiniMax-M2.7)…  风格: "${style}"`);

    try {
        const resp = await fetchJSON(`/api/music/${currentMusicId}/choreo/ai`, 'POST',
                                     { style });
        await pollTask(resp.task_id, '编排生成');

        // 重新加载编排列表（含新生成的）
        await loadChoreoList(currentMusicId);

        // 切换到最新（刚生成）的编排
        const listData = await fetchJSON(`/api/music/${currentMusicId}/choreos`);
        const all = listData.choreos || [];
        if (all.length > 0) {
            const newest = all[all.length - 1];
            setActiveChoreo(newest.choreo_id, newest.theme);
            if (newest.theme) await applyTheme(newest.theme);
            showTask(`✅ 编排完成  来源: ${newest.source || 'ai'}  事件: ${newest.track_count}  风格: "${style}"`);
        } else {
            showTask('✅ 编排完成');
        }
        loadMusicLibrary();
    } catch (err) {
        showTask('❌ 编排失败: ' + err.message);
    } finally {
        document.getElementById('choreBtn').disabled = false;
    }
}

// ── 预览 ────────────────────────────────────────────────────────────
async function togglePreview() {
    if (!currentMusicId) return;
    const btn = document.getElementById('previewBtn');
    try {
        await fetchJSON('/api/session/start', 'POST', {
            music_id:  currentMusicId,
            choreo_id: currentChoreoId || undefined,   // 指定当前选中的编排
            mode:      'rhythm',
        });
        btn.textContent = '⏹ 停止预览';
        btn.onclick = stopPreview;
        document.getElementById('currentFx').textContent = '运行中';

        const audio = document.getElementById('audioPlayer');
        audio.src = `/api/music/${currentMusicId}/audio`;
        audio.currentTime = 0;
        audio.play().catch(e => console.warn('audio play failed:', e));
    } catch (err) {
        showTask('❌ 预览失败: ' + err.message);
    }
}

async function stopPreview() {
    try { await fetchJSON('/api/session/stop', 'POST'); } catch {}
    const audio = document.getElementById('audioPlayer');
    audio.pause();
    audio.src = '';
    const btn = document.getElementById('previewBtn');
    btn.textContent = '▶ 开始预览';
    btn.onclick = togglePreview;
    document.getElementById('currentFx').textContent = '-';
    document.getElementById('audioTime').textContent = '0.0s';
}

// ── 重置 ────────────────────────────────────────────────────────────
function reset() {
    currentMusicId  = null;
    currentChoreoId = null;
    const audio = document.getElementById('audioPlayer');
    audio.pause();
    audio.src = '';
    document.getElementById('musicFile').value = '';
    document.getElementById('fileInfo').classList.add('hidden');
    document.getElementById('uploadProgress').classList.add('hidden');
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('choreBtn').disabled = true;
    document.getElementById('previewBtn').disabled = true;
    document.getElementById('previewBtn').textContent = '▶ 开始预览';
    document.getElementById('previewBtn').onclick = togglePreview;
    document.getElementById('choreoSummary').style.display = 'none';
    document.getElementById('choreoList').innerHTML = '';
    document.getElementById('taskSection').style.display = 'none';
    document.getElementById('beatInfo').textContent = '-';
    document.getElementById('currentFx').textContent = '-';
    document.getElementById('audioTime').textContent = '0.0s';
    document.getElementById('fxActive').classList.add('hidden');
    document.getElementById('themeActive').classList.add('hidden');
    document.getElementById('musicLibrarySelect').value = '';
    document.getElementById('themeSelect').value = '';
    _activeTheme = '';
    applyTheme('');
}

// ── 任务轮询 ────────────────────────────────────────────────────────
function pollTask(taskId, label) {
    return new Promise((resolve, reject) => {
        const timer = setInterval(async () => {
            try {
                const t = await fetchJSON(`/api/task/${taskId}`);
                showTask(`${label}: ${t.status}`);
                if (t.status === 'done') { clearInterval(timer); resolve(); }
                else if (t.status && t.status.startsWith('error')) {
                    clearInterval(timer);
                    reject(new Error(t.status));
                }
            } catch (e) { clearInterval(timer); reject(e); }
        }, 1000);
    });
}

function showTask(msg) {
    const sec = document.getElementById('taskSection');
    sec.style.display = 'block';
    document.getElementById('taskStatus').textContent = msg;
}

// ── WebSocket ───────────────────────────────────────────────────────
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    wsConnection = new WebSocket(`${protocol}//${location.host}/ws/control`);

    wsConnection.onopen  = () => setWsDot(true);
    wsConnection.onmessage = e => {
        try {
            const d = JSON.parse(e.data);
            if (d.subjects !== undefined)
                document.getElementById('subjectCount').textContent = '人数: ' + d.subjects;
            if (d.fx) {
                document.getElementById('currentFx').textContent = d.fx;
                const fa = document.getElementById('fxActive');
                fa.classList.remove('hidden');
                fa.textContent = 'FX: ' + d.fx;
            }
            if (d.theme !== undefined) updateThemeBadge(d.theme);
            if (d.audio_t !== undefined) {
                const audio = document.getElementById('audioPlayer');
                if (audio.paused)
                    document.getElementById('audioTime').textContent = d.audio_t.toFixed(1) + 's';
            }
        } catch {}
    };
    wsConnection.onclose = () => { setWsDot(false); setTimeout(connectWebSocket, 3000); };
    wsConnection.onerror = () => setWsDot(false);
}

function setWsDot(connected) {
    document.getElementById('wsIndicator').className = 'ws-dot ' + (connected ? 'connected' : 'disconnected');
    document.getElementById('wsLabel').textContent   = connected ? 'WS 已连接' : 'WS 未连接';
}

// ── FX 测试区 ───────────────────────────────────────────────────────
let _fxTestActive = {};   // fx_id → intensity

const _FX_CATEGORY_LABEL = {
    overlay:       '覆层特效',
    color:         '颜色调校',
    shader_glitch: '故障特效',
    shader_pixel:  '画质特效',
    digital_ptz:   '运镜',
    time:          '时间域',
    geometry:      '几何变形',
    ai_aware:      'AI感知',
    stylization:   '艺术风格',
    theme:         '主题调色',
};

async function loadFxCatalog() {
    try {
        const data = await fetchJSON('/api/fx/catalog');
        const sel  = document.getElementById('fxTestSelect');
        sel.innerHTML = '<option value="">── 选择特效 ──</option>';

        // 按 category 分组
        const groups = {};
        for (const fx of (data.catalog || [])) {
            if (!groups[fx.category]) groups[fx.category] = [];
            groups[fx.category].push(fx);
        }

        const catOrder = ['overlay','color','shader_glitch','shader_pixel',
                          'digital_ptz','time','geometry','ai_aware','stylization','theme'];
        for (const cat of catOrder) {
            if (!groups[cat]) continue;
            const grp = document.createElement('optgroup');
            grp.label = _FX_CATEGORY_LABEL[cat] || cat;
            for (const fx of groups[cat]) {
                const desc = fx.description ? fx.description.slice(0, 28) : '';
                grp.appendChild(new Option(`${fx.id}  ${desc}`, fx.id));
            }
            sel.appendChild(grp);
        }
    } catch (e) {
        console.warn('loadFxCatalog:', e);
    }
}

async function fxTestToggle() {
    const fxId = document.getElementById('fxTestSelect').value;
    if (!fxId) return;
    const intensity = parseInt(document.getElementById('fxTestIntensity').value, 10) / 100;

    if (_fxTestActive[fxId] !== undefined) {
        // 已激活 → 关闭
        await fxTestRemove(fxId);
    } else {
        // 未激活 → 开启
        try {
            await fetchJSON('/api/session/fx_test', 'POST', { fx_id: fxId, intensity });
            _fxTestActive[fxId] = intensity;
            renderFxChips();
        } catch (e) {
            console.warn('fx_test on:', e);
        }
    }
}

async function fxTestRemove(fxId) {
    try {
        await fetch(`/api/session/fx_test/${encodeURIComponent(fxId)}`,
                    { method: 'DELETE' });
        delete _fxTestActive[fxId];
        renderFxChips();
    } catch (e) {
        console.warn('fx_test off:', e);
    }
}

async function fxTestClearAll() {
    try {
        await fetchJSON('/api/session/fx_test/clear', 'POST');
        _fxTestActive = {};
        renderFxChips();
    } catch (e) {
        console.warn('fx_test clear:', e);
    }
}

function renderFxChips() {
    const container = document.getElementById('fxTestActive');
    container.innerHTML = '';

    // 更新激活按钮文字
    const fxId = document.getElementById('fxTestSelect').value;
    const btn   = container.previousElementSibling;   // 激活按钮
    if (fxId && _fxTestActive[fxId] !== undefined) {
        btn.textContent = '⏹ 关闭';
        btn.className   = 'btn w-full';
    } else {
        btn.textContent = '▶ 激活';
        btn.className   = 'btn btn-accent w-full';
    }

    for (const [id, intens] of Object.entries(_fxTestActive)) {
        const chip = document.createElement('div');
        chip.className = 'fx-chip';
        chip.innerHTML = `<span>${id}</span>
            <span class="chip-intensity">${Math.round(intens * 100)}%</span>
            <button class="chip-remove" onclick="fxTestRemove('${id}')">✕</button>`;
        container.appendChild(chip);
    }
}

// ── 常驻效果控制 ────────────────────────────────────────────────────
let _liveFxDebounce = null;

function onLiveFxSlider(slider, valId) {
    document.getElementById(valId).textContent = slider.value + '%';
    scheduleLiveFxUpdate();
}

function onLiveFxToggle() {
    scheduleLiveFxUpdate();
}

function scheduleLiveFxUpdate() {
    clearTimeout(_liveFxDebounce);
    _liveFxDebounce = setTimeout(pushLiveFxConfig, 80);
}

async function pushLiveFxConfig() {
    const get = id => document.getElementById(id);
    const body = {
        subject_zoom:     get('lfSubjectZoom').checked,
        subject_zoom_amp: parseInt(get('lfSubjectZoomAmp').value, 10) / 100,
        zoom_pulse:       get('lfZoomPulse').checked,
        zoom_pulse_amp:   parseInt(get('lfZoomPulseAmp').value, 10) / 100,
        brightness:       get('lfBrightness').checked,
        brightness_amp:   parseInt(get('lfBrightnessAmp').value, 10) / 100,
    };
    try {
        await fetchJSON('/api/session/live_fx', 'POST', body);
    } catch (e) {
        console.warn('live_fx update failed:', e);
    }
}

// ── 工具函数 ────────────────────────────────────────────────────────
async function fetchJSON(url, method = 'GET', body = null) {
    const opts = { method, headers: {} };
    if (body) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    }
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(`HTTP ${r.status} ${await r.text()}`);
    return r.json();
}
