// PicoClaw 前端 App.js
let currentMusicId = null;
let wsConnection   = null;
let _taskPollTimer = null;
let _choreoTaskId  = null;
let _activeTheme   = '';

// ── 初始化 ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    connectWebSocket();
    checkSystemStatus();
    loadMusicLibrary();
    loadThemeCatalog();
    setInterval(checkSystemStatus, 5000);

    // 音频时间更新（本地时钟显示 + 向服务端同步）
    const audio = document.getElementById('audioPlayer');
    audio.addEventListener('timeupdate', () => {
        document.getElementById('audioTime').textContent =
            audio.currentTime.toFixed(1) + 's';
    });
    // 每 200ms 把 audio.currentTime 同步给服务端（驱动 FX 时钟）
    setInterval(() => {
        if (!audio.paused && wsConnection &&
            wsConnection.readyState === WebSocket.OPEN) {
            wsConnection.send(JSON.stringify({
                type: 'audio_clock', t: audio.currentTime
            }));
        }
    }, 200);

    // 主题强度滑块标签同步
    const slider = document.getElementById('themeStrength');
    slider.addEventListener('input', () => {
        document.getElementById('themeStrengthVal').textContent =
            slider.value + '%';
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
    // 服务器还没就绪时，1s 后重连
    setTimeout(() => { img.src = '/api/video/stream?' + Date.now(); }, 1000);
}

// ── 音乐库 ──────────────────────────────────────────────────────────
async function loadMusicLibrary() {
    const hint = document.getElementById('libraryHint');
    try {
        const data = await fetchJSON('/api/music/library');
        const sel  = document.getElementById('musicLibrarySelect');
        // 保留第一个空选项
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
    currentMusicId = musicId;

    // 清理旧状态
    document.getElementById('choreoSummary').style.display = 'none';
    document.getElementById('taskSection').style.display   = 'none';
    document.getElementById('uploadProgress').classList.add('hidden');

    try {
        const info = await fetchJSON(`/api/music/${musicId}`);
        document.getElementById('fileInfo').classList.remove('hidden');
        document.getElementById('fileName').textContent =
            info.filename || musicId;
        document.getElementById('fileDuration').textContent =
            info.duration ? info.duration.toFixed(1) + 's' : '-';
        document.getElementById('beatInfo').textContent =
            info.tempo ? info.tempo.toFixed(0) + ' BPM' : '-';

        // 按状态决定哪些按钮可用
        const analyzed = info.status === 'analyzed';
        document.getElementById('analyzeBtn').disabled  = false;
        document.getElementById('choreBtn').disabled    = !analyzed;
        document.getElementById('previewBtn').disabled  = true;

        // 尝试加载编排
        try {
            const choreo = await fetchJSON(`/api/music/${musicId}/choreo`);
            showChoreoResult(choreo);
            document.getElementById('previewBtn').disabled = false;
            // 自动切换到编排中记录的主题
            if (choreo.theme) {
                document.getElementById('themeSelect').value = choreo.theme;
                _activeTheme = choreo.theme;
                updateThemeBadge(choreo.theme);
            }
        } catch {
            // 无编排——正常，不报错
        }
    } catch (e) {
        showTask('❌ 加载音乐信息失败: ' + e.message);
    }
}

// ── 主题滤镜 ────────────────────────────────────────────────────────
async function loadThemeCatalog() {
    try {
        const data = await fetchJSON('/api/fx/themes');
        const sel  = document.getElementById('themeSelect');
        for (const t of (data.themes || [])) {
            const label = t.description
                ? `${t.id}  —  ${t.description}`
                : t.id;
            sel.add(new Option(label, t.id));
        }
    } catch (e) {
        console.warn('loadThemeCatalog:', e);
    }
}

async function applyTheme(themeId) {
    const strength = parseInt(document.getElementById('themeStrength').value, 10) / 100;
    document.getElementById('themeStrengthVal').textContent =
        Math.round(strength * 100) + '%';
    try {
        await fetchJSON('/api/session/theme', 'POST',
                        { theme_id: themeId || '', strength });
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

    // 读取时长
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const reader = new FileReader();
        reader.onload = ev => ctx.decodeAudioData(ev.target.result, buf => {
            const m = Math.floor(buf.duration / 60);
            const s = Math.floor(buf.duration % 60);
            document.getElementById('fileDuration').textContent = `${m}:${String(s).padStart(2,'0')}`;
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
                    console.log('上传成功, music_id:', currentMusicId);
                    // 刷新音乐库列表
                    loadMusicLibrary();
                } else {
                    document.getElementById('progressText').textContent =
                        `❌ 上传失败 (HTTP ${xhr.status})`;
                    console.error('上传失败:', xhr.status, xhr.responseText);
                }
            } catch (e) {
                document.getElementById('progressText').textContent = '❌ 响应解析失败: ' + e.message;
                console.error('upload parse error:', e, xhr.responseText);
            }
            resolve();
        });
        xhr.addEventListener('error', () => {
            document.getElementById('progressText').textContent = '❌ 网络错误，请检查连接';
            resolve();
        });
        xhr.addEventListener('timeout', () => {
            document.getElementById('progressText').textContent = '❌ 上传超时';
            resolve();
        });
        xhr.timeout = 120000;  // 120s 超时（原始流比 multipart 快 37x）
        xhr.open('PUT', '/api/music/upload');
        xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');
        xhr.send(file);  // 直接发送原始文件，无 multipart 开销
    });
}

// ── 分析节奏 ────────────────────────────────────────────────────────
async function analyzMusic() {
    if (!currentMusicId) return;
    document.getElementById('analyzeBtn').disabled = true;
    showTask('节奏分析中...');

    try {
        const resp = await fetchJSON(`/api/music/${currentMusicId}/analyze`, 'POST');
        const taskId = resp.task_id;
        await pollTask(taskId, '节奏分析');

        const info = await fetchJSON(`/api/music/${currentMusicId}`);
        showTask(`✅ 分析完成  BPM: ${info.tempo ? info.tempo.toFixed(1) : '?'}  时长: ${info.duration ? info.duration.toFixed(1)+'s' : '?'}  节拍: ${info.beat_count}`);
        document.getElementById('beatInfo').textContent = info.tempo ? info.tempo.toFixed(0) + ' BPM' : '-';
        document.getElementById('choreBtn').disabled = false;
        loadMusicLibrary();  // 刷新库（显示已分析状态）
    } catch (err) {
        showTask('❌ 分析失败: ' + err.message);
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// ── 生成编排 ────────────────────────────────────────────────────────
async function generateChoreo() {
    if (!currentMusicId) return;
    document.getElementById('choreBtn').disabled = true;
    showTask('AI 编排生成中 (MiniMax-M2.7)...');

    try {
        const resp = await fetchJSON(`/api/music/${currentMusicId}/choreo/ai`, 'POST',
                                     { style: 'energetic' });
        _choreoTaskId = resp.task_id;
        await pollTask(_choreoTaskId, '编排生成');

        const choreo = await fetchJSON(`/api/music/${currentMusicId}/choreo`);
        showTask(`✅ 编排完成  来源: ${choreo.source}  事件: ${choreo.track_count}`);
        showChoreoResult(choreo);
        document.getElementById('previewBtn').disabled = false;
        // 自动应用编排中的主题
        if (choreo.theme) {
            document.getElementById('themeSelect').value = choreo.theme;
            await applyTheme(choreo.theme);
        }
        loadMusicLibrary();  // 刷新库（显示 ✓ 标记）
    } catch (err) {
        showTask('❌ 编排失败: ' + err.message);
        document.getElementById('choreBtn').disabled = false;
    }
}

function showChoreoResult(choreo) {
    const sec = document.getElementById('choreoSummary');
    const info = document.getElementById('choreoInfo');
    sec.style.display = 'block';

    let plan = choreo.plan || {};
    let lines = [
        `来源: ${choreo.source || '-'}`,
        `风格: ${choreo.style || '-'}`,
        `事件数: ${choreo.track_count}`,
    ];
    if (choreo.theme) lines.push(`主题: ${choreo.theme}`);
    if (plan.segments) lines.push(`段落: ${plan.segments.length}`);
    if (plan.mood)     lines.push(`情绪: ${plan.mood}`);
    info.innerHTML = lines.map(l => `<div>${l}</div>`).join('');

    // 在视频 overlay 显示激活特效
    const fxEl = document.getElementById('fxActive');
    fxEl.classList.remove('hidden');
    fxEl.textContent = 'AI编排已加载';
}

// ── 预览 ────────────────────────────────────────────────────────────
async function togglePreview() {
    if (!currentMusicId) return;
    const btn = document.getElementById('previewBtn');
    try {
        await fetchJSON('/api/session/start', 'POST', {
            music_id: currentMusicId, mode: 'rhythm'
        });
        btn.textContent = '⏹ 停止预览';
        btn.onclick = stopPreview;
        document.getElementById('currentFx').textContent = '运行中';

        // 播放音乐（用服务端音频流，浏览器解码）
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
    currentMusicId = null;
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
    document.getElementById('taskSection').style.display = 'none';
    document.getElementById('beatInfo').textContent = '-';
    document.getElementById('currentFx').textContent = '-';
    document.getElementById('audioTime').textContent = '0.0s';
    document.getElementById('fxActive').classList.add('hidden');
    document.getElementById('themeActive').classList.add('hidden');
    document.getElementById('musicLibrarySelect').value = '';
    // 清除主题
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

    wsConnection.onopen = () => {
        setWsDot(true);
    };
    wsConnection.onmessage = e => {
        try {
            const d = JSON.parse(e.data);
            // 更新人数
            if (d.subjects !== undefined) {
                document.getElementById('subjectCount').textContent = '人数: ' + d.subjects;
            }
            // 当前特效
            if (d.fx) {
                document.getElementById('currentFx').textContent = d.fx;
                const fa = document.getElementById('fxActive');
                fa.classList.remove('hidden');
                fa.textContent = 'FX: ' + d.fx;
            }
            // 主题
            if (d.theme !== undefined) {
                updateThemeBadge(d.theme);
            }
            // 音频时间（若本地 audio 在播放则忽略服务端时钟，避免抖动）
            if (d.audio_t !== undefined) {
                const audio = document.getElementById('audioPlayer');
                if (audio.paused) {
                    document.getElementById('audioTime').textContent = d.audio_t.toFixed(1) + 's';
                }
            }
        } catch {}
    };
    wsConnection.onclose = () => {
        setWsDot(false);
        setTimeout(connectWebSocket, 3000);
    };
    wsConnection.onerror = () => setWsDot(false);
}

function setWsDot(connected) {
    const dot   = document.getElementById('wsIndicator');
    const label = document.getElementById('wsLabel');
    dot.className   = 'ws-dot ' + (connected ? 'connected' : 'disconnected');
    label.textContent = connected ? 'WS 已连接' : 'WS 未连接';
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
