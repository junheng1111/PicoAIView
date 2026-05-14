# PicoClaw on 地瓜机器人 RDK X5 (BPU) — 设计文档（纯软件 + LLM 编排版）

> 目标：在 **地瓜机器人 RDK X5** 上跑 PicoClaw 服务；上传一段音乐，**摄像头画面**根据音乐节奏产生视觉律动，由 RDK X5 内置 **BPU** 提供"看懂画面"的能力，并由 **LLM Choreographer Agent**（云端 Claude / GPT）担任"导演"，根据音乐特征从特效库挑选并编排镜头语言（**纯数字效果，不涉及任何机械云台/舵机**）。
>
> RDK X5 BPU 提供 **10 TOPS** AI 算力，足以在 1080p@30fps 下并行跑人体检测 + 姿态估计 + 分割。LLM Agent 把原本硬编码的 `build_choreo()` 规则升级为"按音乐风格动态决策的导演脑"，是本系统从"节拍 → 固定特效"跨越到"节拍 → 智能镜头编排"的关键。

---

## 1. 项目目标与边界

| 项 | 说明 |
| --- | --- |
| ✅ 范围 | 摄像头采集 → 软件特效（数字 pan/zoom、滤镜、抖动、亮度、叠加、转场） → 跟随音乐节奏 → WebRTC/HLS 推给前端 |
| ✅ 范围 | 音乐上传、节奏分析、编排预览、播放同步 |
| ✅ 范围 | **LLM 导演 Agent：上传后调用云端 Claude / GPT，基于音频特征 + FX 库目录自动产出 ChoreoPlan（段落级镜头方案）** |
| ❌ 不做 | 物理云台、舵机、电机、灯带等任何硬件控制 |
| ❌ 不做 | 多机/集群、支付、账号体系 |
| ❌ 不做 | 让 LLM 实时逐帧出指令（成本/延迟不可控，改为离线段落级规划） |

---

## 2. 硬件最小集

| 硬件 | 说明 |
| --- | --- |
| **地瓜机器人 RDK X5** | 八核 Cortex-A55，内置 10 TOPS BPU，无需外接加速卡 |
| **RDK X5 内置 BPU** | 10 TOPS，跑 YOLOv5/YOLOv8/姿态/分割（.bin 格式模型） |
| MIPI CSI 摄像头（RDK X5 官方摄像头或兼容模组） | 低延迟，配合 hobot_sensor + v4l2 最顺 |
| USB 声卡或 HDMI 音频（可选） | 浏览器侧播放音频时可省 |
| NVMe SSD 或高速 TF 卡（可选） | 长时间推理 + 模型/素材读取 |

> **预装环境**：Ubuntu 22.04（RDK X5 官方镜像）+ `hobot-dnn` 一键包（含 BPU 驱动、horizon_dnn、模型库、GStreamer 插件 `hobot_codec`）。安装后 `hrt_model_exec model --model_file=xxx.bin` 可验证 BPU 推理正常。

---

## 3. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│                浏览器 / 移动端 Web 客户端                    │
│  上传音乐 · 控制台 · 视频画面 (WebRTC) · 同步播放音频        │
└──────────────┬───────────────────────────────────────────────┘
               │ HTTPS / WebSocket / WebRTC
┌──────────────▼───────────────────────────────────────────────┐
│  地瓜机器人 RDK X5  (Ubuntu 22.04, Python 3.12, asyncio)     │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ FastAPI  │  │ BeatEngine   │  │ MediaPipeline          │ │
│  │ + WS     │─▶│ librosa 离线 │  │ MIPI Camera / v4l2  │ │
│  │ 控制 API │  │ tempo/beats  │  │       │                │ │
│  └────┬─────┘  │ onsets/RMS   │  │       ▼                │ │
│       │        │ + segments   │  │  ┌──────────────────┐  │ │
│       │        └──────┬───────┘  │  │ Vision Engine    │◀─┼─┼─ 片上BPU ─ ┌──────────┐
│       │               │          │  │ (RDK X5 BPU)     │  │ │            │  BPU     │
│       │               ▼          │  │ YOLO 人/物 检测  │  │ │            │ 10 TOPS  │
│       │      ┌─────────────────┐ │  │ Pose 姿态        │  │ │            └──────────┘
│       │      │ AI Choreographer│ │  │ Segment 人像分割 │  │ │
│       │      │ (Claude/GPT API)│ │  │ → tracks/keypts  │  │ │
│       │      │ feat + FX 目录  │ │  └────────┬─────────┘  │ │
│       │      │  → ChoreoPlan   │ │           ▼            │ │
│       │      └────────┬────────┘ │  FX Compositor         │ │
│       │               │          │  (Plan + 节拍 + 视觉)  │ │
│       │               │          │           ▼            │ │
│       │               │          │  HW H.264 → aiortc     │ │
│       │               │          └────────────┬───────────┘ │
│       ▼               ▼                       ▼             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │       Orchestrator (asyncio EventBus)                  │ │
│  │ beat.tick / onset / energy / vision.subject /          │ │
│  │ vision.pose / fx.cmd / plan.segment / clock.sync       │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**核心思想（RDK X5 BPU + LLM Agent 双引擎）**：
1. **音频侧**：librosa 离线分析音乐 → 节拍/重拍/onset/RMS + 自动段落切分（intro/verse/chorus/bridge/outro）。
2. **导演侧**：**LLM Choreographer Agent** 接到音频特征摘要 + 当前 FX 库目录后，一次性输出整首曲目的 **ChoreoPlan**（段落级镜头方案，例如"副歌用 zoom_pulse + ai_subject_track + saturation_flash"）。
3. **视觉侧**：RDK X5 BPU 实时给出**主体在哪里、人在做什么动作、人像 mask**。
4. **FX Compositor**：把 ① ChoreoPlan（段落策略）+ ② 节拍时间轴 + ③ 视觉理解 三层融合成最终画面。
   - 例：LLM 在 chorus1 段写了 `ai_subject_track`，runtime 用 Hailo 的 primary subject 实时填补"跟谁"；LLM 写了 `zoom_pulse`，runtime 用 BeatEngine 的 downbeat 触发。

---

## 4. 数字"律动"特效库（Hailo 优先 · 低延迟版）

> 设计哲学（v0.4.1 起）：
> 1. **重点 = RDK X5 BPU 驱动的 AI-aware 特效**（这是本机相比"普通开发板"的护城河，LLM Choreographer 在副歌/桥段/转场会优先选这一类）。
> 2. **每帧 FX 链路严格预算 ≤ 12 ms**，配合 H.264 硬编保证端到端 < 250ms。
> 3. **优先 GPU shader（Mali GPU / OpenGL ES）实现** —— 大量 fx 都能压到 <2ms / pass，可叠加而不爆预算。
> 4. **BPU 推理永远在独立线程**，FX Compositor 只读 VisionState 最近一帧，**推理慢不阻塞 FX 链路**。

### 4.1 延迟预算（每帧 33ms @ 30fps）

| 阶段 | 预算 | 实现 |
| --- | --- | --- |
| 摄像头采集 + zero-copy | ~3 ms | hobot_sensor/v4l2 |
| BPU 视觉推理 | **异步**，不阻塞 FX | 独立线程，FX 用 VisionState 最近帧 |
| **FX 合成链（≤4 个 fx 叠加）** | **≤ 12 ms** | shader 优先，OpenCV 兜底 |
| H.264 硬编 | ~5 ms | hobot_codec / v4l2 m2m |
| WebRTC 网络 | ~150 ms | aiortc |

**单 fx 延迟分级：**
- 🟢 **轻量 < 2ms**：可任意叠加（一段挂 4 个都行）
- 🟡 **中等 2–5ms**：一段最多 3 个
- 🔴 **重型 5–12ms**：独占，不与其他 🔴 同段

> 这套分级会被 LLM Choreographer 当作硬约束（见 §5.3.3）。

---

### 4.2 BPU 驱动 AI-aware 特效（⭐ 重点）

> 这是 LLM Choreographer 的"招牌库"。每个 fx 的"AI 感知部分"在 RDK X5 BPU 上独立线程跑，**FX Compositor 只消费 VisionState 指针**，本体延迟仅来自合成步骤本身。
>
> RDK X5 BPU 上 yolov5/yolov8/pose/seg/depth 推理实测：**单模型 20–40ms**，靠多模型上下文切换或并发，主路径 **30fps 不掉帧**。FX 用最近一帧推理结果（即"视觉滞后 ≤ 1 帧 = 33ms"），人耳人眼基本感知不到与音乐错位。

| 特效 | 依赖模型 | FX 本体延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ 智能跟拍 (`subject_track`) | yolov8s | 🟢 ~1ms（裁剪） | 副歌主拍、持续 |
| ✅ 节拍 bbox (`beat_bbox`) | yolov8s | 🟢 <1ms（描边） | 主拍 |
| ✅ 姿态光剑 (`pose_trail`) | yolov8s_pose | 🟡 ~3ms（关键点拖尾） | onset 爆发 |
| ✅ 人像滤镜 (`portrait_filter`) | yolov8s_seg | 🟡 ~4ms（mask 加权混合） | 副歌 |
| ✅ 背景替换 (`bg_replace`) | yolov8s_seg | 🟡 ~5ms | 段落切换 |
| ✅ 舞步触发 (`motion_react`) | yolov8s_pose | 🟢 <1ms（仅判定） | 实时（驱动其他 fx） |
| ✅ 多人聚焦 (`multi_person_focus`) | yolov8s + tracker | 🟢 ~1ms | downbeat 切主角 |
| 🆕 **景深虚化 (`depth_dof`)** | fast_depth | 🟡 ~5ms（分层 blur） | 桥段、抒情段（电影感） |
| 🆕 **伪 3D 视差 (`pseudo_3d_parallax`)** | fast_depth | 🟡 ~4ms | 桥段镜头推进 |
| 🆕 **语义背景模糊 (`semantic_bg_blur`)** | yolov8s_seg | 🟡 ~3ms | minimal 风、人像突出 |
| 🆕 **粒子跟手势 (`pose_particles`)** | yolov8s_pose | 🟡 ~3ms（粒子发射 from 关键点） | drop |
| 🆕 **人群密度反应 (`crowd_density_react`)** | yolov8s | 🟢 <1ms（元数据） | 全段持续，驱动其他 fx 强度 |
| 🆕 **双人镜像 (`multi_person_mirror`)** | yolov8s + tracker | 🟡 ~4ms | 双人舞 |
| 🆕 **脸部贴纸 (`face_filter`)** | scrfd_2.5g | 🟡 ~3ms（仿射 + 叠加） | 卡拉 OK 段 |
| 🆕 **主体形状 match cut (`subject_match_cut`)** | yolov8s_seg | 🟡 ~4ms（仅转场瞬间） | downbeat 高级转场 |

**BPU 资源排程**（避免 NPU 过热 / 抢占）：
- 默认常驻：`yolov8s`（检测）+ `yolov8s_pose`（姿态）—— 两个并发跑 30fps
- 副歌段按需切入 `yolov8s_seg`（分割）—— 抢占式，分割段限 ≤60s 连续，否则降到 15fps
- `fast_depth` 在桥段才加载（context switch ~50ms，预热提前 1s）
- `scrfd_2.5g` 仅卡拉 OK 段加载，且互斥不与分割同跑

---

### 4.3 GPU Shader 类（廉价，可大量叠加）

> RDK X5 Mali GPU 上跑，集成方式：`gst-glshader` GStreamer element 或 `moderngl` Python 绑定。**典型 0.5–2ms / pass**，多 pass 可串联。

#### 4.3.1 像素级
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ 亮度脉冲 / 饱和爆闪 / 色相旋转 / 反色 / 单色 | shader | 🟢 <1ms | 已有 |
| 🆕 **Bloom / Glow** | shader（亮度阈值 + 高斯 + add） | 🟡 ~3ms | 副歌爆点 |
| 🆕 **Vignette（暗角）** | shader | 🟢 <1ms | 聚焦感 |
| 🆕 **Film Grain（胶片噪点）** | shader（伪随机 hash） | 🟢 <1ms | 复古 MV |
| 🆕 **Chromatic Aberration（RGB 错位）** | shader | 🟢 ~1ms | drop / glitch |
| 🆕 **Posterize / Bit-crush（色阶减少）** | shader | 🟢 <1ms | 8-bit 风 |
| 🆕 **Halftone（漫画网点）** | shader | 🟢 ~1ms | 转场、桥段 |
| 🆕 **Pixelate / Mosaic** | shader（block 平均） | 🟢 ~1ms | 静音段反差 |
| 🆕 **Motion Blur（方向模糊）** | shader（速度向量） | 🟡 ~2ms | 加速感、whip pan |

#### 4.3.2 几何变形
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ Pan / Tilt / Zoom / Shake | shader（仿射） | 🟢 <1ms | 已有 |
| ✅ 鱼眼 / 镜像分屏 / 万花筒 | shader | 🟢 ~1ms | 已有 |
| 🆕 **Wave / Ripple（水波）** | shader（sin displacement） | 🟢 ~1ms | 桥段、回忆 |
| 🆕 **Twirl / Swirl（漩涡）** | shader（极坐标） | 🟢 ~1ms | 转场 |
| 🆕 **Tunnel / Polar Warp** | shader | 🟢 ~1ms | 副歌迷幻 |
| 🆕 **Whip Pan（甩镜转场）** | shader（pan + 方向 blur） | 🟡 ~2ms | downbeat 转场 |

#### 4.3.3 调色（电影级，性价比 No.1）⭐
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| 🆕 **3D LUT（.cube 文件）** ⭐ | shader（`texture(lut3d, color)`） | 🟢 <1ms | **每段套不同 LUT，电影感立刻拉满** |
| 🆕 **Curves（RGB 曲线）** | shader（1D LUT） | 🟢 <1ms | 暖/冷段 |
| 🆕 **Duotone（双色调）** | shader（gradient map） | 🟢 <1ms | minimal 风 |
| 🆕 **Selective Color** | shader（HSV mask） | 🟢 ~1ms | 仅某色变化 |
| 🆕 **Color Cycling** | shader（hue offset 动画） | 🟢 <1ms | 高 BPM 段 |

#### 4.3.4 故障美学（glitch）⭐ MV 必备
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ Glitch Cut（已有，可大幅扩） | shader | 🟢 ~1ms | 转场 |
| 🆕 **VHS Tracking Error（录像带卷曲）** | shader | 🟡 ~2ms | 复古 / 赛博 |
| 🆕 **Scanlines / CRT** | shader | 🟢 <1ms | 8-bit / 赛博 |
| 🆕 **Datamosh（伪数据马赛克）** | shader（块向量混乱） | 🟡 ~2ms | drop 瞬间 |
| 🆕 **JPEG Compression Artifacts** | shader（DCT 量化） | 🟡 ~2ms | drop |
| 🆕 **Roll Bar（行同步错位）** | shader | 🟢 ~1ms | 故意"坏"画面 |
| 🆕 **RGB Shift Strobe** | shader | 🟢 ~1ms | 高 BPM 段 |

---

### 4.4 时间域（需帧缓冲）
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ 帧抖动定格 / 慢动作 / Strobe | 帧缓冲 | 🟢 <1ms | 已有 |
| 🆕 **Echo / Trails（残影拖尾）** | 前 N 帧 alpha 叠加（GPU） | 🟡 ~2ms | rave / electronic drop |
| 🆕 **Slit-scan** | 行级帧混合 | 🔴 ~6ms | 实验段（独占） |

> 时间域需维护最近 8 帧 ring buffer，内存约 +50MB；不在采集帧之外再开新延迟。

---

### 4.5 光效粒子（叠加层）
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ 节拍光圈 / 频谱条 / 粒子 / 歌词 | 叠加层 | 🟢 ~1ms | 已有 |
| 🆕 **Lens Flare（镜头光晕）** | shader（光源点 + 光斑） | 🟡 ~2ms | 重拍点睛 |
| 🆕 **Light Leak（漏光）** | 预渲染 PNG 序列 + screen 混合 | 🟢 <1ms | 复古 / 温暖段 |
| 🆕 **God Rays（体积光）** | shader（径向 blur from 光源） | 🟡 ~3ms | 副歌情绪 |
| 🆕 **Confetti / Snow / Spark 粒子** | CPU 粒子 + GPU 合成 | 🟡 ~2ms | 副歌 |
| 🆕 **音频反应几何（FFT-driven 多边形）** | shader uniforms | 🟢 ~1ms | 实时能量驱动 |

---

### 4.6 风格化渲染（重型，谨慎使用）
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| 🆕 **Cartoon / Cel-shading** | shader（edge + posterize） | 🔴 ~7ms | 副歌反差（独占） |
| 🆕 **Pencil Sketch** | OpenCV `pencilSketch()` | 🔴 ~6ms | 纯净段（独占） |
| 🆕 **Oil Painting** | OpenCV xphoto | 🔴 ~10ms | 桥段（独占） |
| ❌ **Neural Style Transfer** | Hailo Model Zoo 暂无成熟 HEF | — | **v0.4.1 不实现**，留 §14 |
| ❌ **Real-time Super-resolution** | 同上 | — | **v0.4.1 不实现** |

> 风格化类一段最多挂 1 个，且不能与其他 🔴 重型同段；LLM prompt 中已注入此互斥约束。

---

### 4.7 转场专项
| 特效 | 实现 | 延迟 | 用途 |
| --- | --- | --- | --- |
| ✅ Cut / Fade | 单帧切换 | 🟢 <1ms | 已有 |
| ✅ Glitch（基础） | shader | 🟢 ~1ms | 已有 |
| 🆕 **Whip Pan / Zoom Blur Transition** | shader | 🟡 ~2ms（仅转场瞬间） | downbeat |
| 🆕 **Wipe（径向 / 几何）** | shader mask | 🟢 ~1ms | 段落切换 |
| 🆕 **Light Leak Transition** | 预渲染叠加 | 🟢 <1ms | 温暖切换 |
| 🆕 **Subject Match Cut** | yolov8s_seg + 形状对齐 | 🟡 ~4ms | LLM 高级选项（downbeat） |

---

### 4.8 LLM Choreographer 选用约束（自动注入到 prompt，见 §5.3.3）

```
【硬约束 — 违反必判 invalid，触发自纠重试】
1. 单段最多挂 4 个 fx，且：≤2 个 🟡，≤1 个 🔴，🟢 不限。
2. 风格化类（cartoon/sketch/oil_painting）每段最多 1 个，与所有 🔴 互斥。
3. yolov8s_seg、fast_depth、scrfd_2.5g 不能在相邻两个段同时用（避免 BPU
   context switch 抖动）；连续使用同一重型模型不超过 60s。
4. v0.4.1 排除清单：style_transfer、super_resolution（HEF 不可用）。

【软偏好 — 仅作 prompt 引导】
1. 副歌 / drop 段：优先选 §4.2 的 Hailo signature fx + 1 个 glitch shader。
2. intro / outro：偏向 §4.3.3 调色（LUT、duotone、vignette）+ 1 个轻几何。
3. bridge：偏向时间域（slow-mo / echo）+ depth_dof，制造反差。
4. 静音段：fx 总数 ≤ 1，避免画面"为效果而效果"。
```

> 所有 fx 参数都暴露为 0~1 标量，由编排轨控制；前端可实时预览/调参。
> ⭐ 标记 = 强烈推荐落地的低延迟高质感组合（3D LUT、Hailo signature、glitch shader）。

---

## 5. 节奏分析（BeatEngine）

### 5.1 离线分析（推荐主路径）

```python
import librosa, numpy as np, json

def analyze(path: str) -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
    onsets     = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    rms        = librosa.feature.rms(y=y)[0]
    times      = librosa.times_like(rms, sr=sr)
    # 估计 downbeat（每 4 拍重拍）
    downbeats  = beats[::4]
    return {
        "tempo": float(tempo),
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "onsets": onsets.tolist(),
        "rms":   list(zip(times.tolist(), rms.tolist())),
    }
```

输出存为 `<music_id>.beats.json`，供运行时播放使用。

> **5.1 增量**：在原有特征之外，再用 `librosa.segment` 自动切分段落（intro/verse/chorus/bridge/outro 等），输出到 `feat["segments"]`。这是 AI Choreographer 决策时的核心上下文。

```python
def detect_segments(y, sr, beats):
    # 用 chroma + recurrence matrix 做结构分割
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    bounds = librosa.segment.agglomerative(chroma, k=None)  # 自动 k
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    # 给每段打能量标签（rms 均值 → low/mid/high）
    segments = []
    for t0, t1 in zip(bound_times[:-1], bound_times[1:]):
        rms_seg = rms_in_range(t0, t1)
        segments.append({"t_start": t0, "t_end": t1, "energy": rms_seg, "label": auto_label(rms_seg)})
    return segments
```

### 5.2 ChoreoTrack 编排生成 — 规则模式（fallback）

> 这是 v0.3 的旧路径，**v0.4 起降级为 fallback**：仅当 LLM API 不可用或返回非法 JSON 时使用。主路径见 §5.3 AI Choreographer。

```python
def build_choreo_rulebased(feat):
    """Deterministic fallback. 不依赖 LLM。"""
    track = []
    for t in feat["beats"]:
        track.append({"t": t, "fx": "pan",  "v": "sin"})       # 主拍轻摆
    for t in feat["downbeats"]:
        track.append({"t": t, "fx": "zoom_pulse", "amp": 0.08})  # 重拍缩放
        track.append({"t": t, "fx": "flash", "amp": 0.4})
    for t in feat["onsets"]:
        track.append({"t": t, "fx": "shake", "amp": 4})
    # RMS → 亮度曲线（连续）
    track.append({"t": 0, "fx": "brightness_curve", "data": feat["rms"]})
    return sorted(track, key=lambda x: x["t"])
```

### 5.3 AI Choreographer — LLM 编排 Agent（主路径）

> **设计动机**：规则版 `build_choreo_rulebased` 把所有歌曲都当成"主拍摆 + 重拍 zoom + onset 抖"的同一个模板，缺乏风格识别。让一个 LLM 充当"导演"，看一眼音乐特征摘要就能输出"intro 用淡入 + 缓慢 pan，drop 用 strobe + 万花筒 + 智能跟拍"这种段落级别的镜头语言。整首歌仅调用 1 次 LLM，离线生成，**运行时零延迟**。

#### 5.3.1 Pipeline

```
   feat (BeatEngine 输出)             FX_REGISTRY (运行时枚举)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
            ┌───────────────────────────┐
            │ build_agent_input(feat)   │  缩减体积，避免 token 爆炸
            │  - bpm, duration          │  （beats 不全发，发段落摘要 + 直方图）
            │  - segments[]             │
            │  - rms_summary[]          │
            │  - fx_catalog (id+desc+   │
            │     params schema)        │
            └─────────────┬─────────────┘
                          ▼
                ┌─────────────────┐
                │  LLM Provider   │  Anthropic / OpenAI
                │  (JSON mode)    │  system prompt = "你是 MV 导演..."
                └────────┬────────┘
                         ▼
                  ChoreoPlan JSON
                         │
                         ▼
            ┌────────────────────────┐
            │ validate_plan(schema)  │ → 失败 → fallback to §5.2
            └─────────────┬──────────┘
                          ▼
            ┌────────────────────────┐
            │ expand_plan_to_track() │ Plan(段落级) → Track(时间点级)
            └─────────────┬──────────┘
                          ▼
                <music_id>.choreo.json
```

#### 5.3.2 LLM Provider 抽象

```python
# app/ai_choreographer/provider.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def complete_json(self, system: str, user: str, schema: dict) -> dict: ...

class ClaudeProvider(LLMProvider):
    """主选。Claude 4.x，JSON mode + 工具调用。"""
    model = "claude-opus-4-7"  # 或 claude-sonnet-4-6 降本

class OpenAIProvider(LLMProvider):
    """备选。GPT-4o/5，response_format=json_schema。"""
```

环境变量：`PICOCLAW_LLM=claude|openai`、`ANTHROPIC_API_KEY=...`、`OPENAI_API_KEY=...`。默认 Claude。

#### 5.3.3 Prompt 设计

**System prompt**：

```
你是一名 MV / Live 视觉导演 AI，名叫 PicoClawDirector。
你的任务：根据一段音乐的客观特征，从给定的特效库中挑选并编排镜头方案。

【硬约束 — 违反则输出会被判 invalid 并重试】
1. 输出必须是合法 JSON，严格符合 ChoreoPlan schema。
2. 只能使用 fx_catalog 中列出的 fx id，禁止编造。
3. params 必须落在 fx_catalog 给出的取值范围内。
4. 单段最多挂 4 个 fx，且延迟分级满足：
   - 🟢 (latency_tier=light)  不限数量
   - 🟡 (latency_tier=medium) ≤ 2 个 / 段
   - 🔴 (latency_tier=heavy)  ≤ 1 个 / 段
5. 风格化类（fx.category=stylization）每段最多 1 个，与所有 🔴 互斥。
6. Hailo 重型推理模型（depends_on ∈ {seg, depth, face_landmark}）：
   - 不能在相邻两段都使用同一模型（避免 NPU context switch 抖动）；
   - 同一重型模型连续使用累计不超过 60 秒。
7. v0.4.1 排除清单（fx_catalog.excluded）中的 fx 不能出现在 plan 中。

【软偏好 — 影响 fx 选择风格】
1. 重点 = BPU 驱动的 AI-aware fx（fx.category=ai_aware），副歌 / drop 段优先选。
2. intro / outro：偏向调色（3d_lut、duotone、vignette）+ 1 个轻几何。
3. bridge：偏向时间域 + depth_dof，制造反差。
4. 静音段（energy=low）：fx 总数 ≤ 1。
5. 用户 style 字段会改变偏好但不能突破硬约束。
```

**User prompt（结构化，约 1.5–3k tokens）**：

```
## 音频特征
- 时长: 222.4s
- BPM: 128
- 主拍数: 472, downbeat 数: 118, onset 平均密度: 4.2/s
- RMS 曲线（每 4s 一格 mean，0-1 归一化）: [0.12, 0.18, 0.45, 0.71, 0.74, ...]
- 段落（自动检测）:
  [{"t":[0,18],"label":"intro","energy":"low"},
   {"t":[18,51],"label":"verse1","energy":"mid"},
   {"t":[51,84],"label":"chorus1","energy":"high"},
   ...]

## 可用特效库 v0.4.1（自动从 FX_REGISTRY 序列化，含延迟分级与 Hailo 依赖）
[
  {"id":"pan",              "category":"digital_ptz",   "latency_tier":"light",  "depends_on":[],          "params":{"amp":[0,0.1]}},
  {"id":"zoom_pulse",       "category":"digital_ptz",   "latency_tier":"light",  "depends_on":[],          "params":{"amp":[0,0.2]}},
  {"id":"shake",            "category":"digital_ptz",   "latency_tier":"light",  "depends_on":[],          "params":{"px":[1,12]}},
  {"id":"saturation_flash", "category":"color",         "latency_tier":"light",  "depends_on":[],          "params":{"amp":[0,1]}},
  {"id":"3d_lut",           "category":"color",         "latency_tier":"light",  "depends_on":[],          "params":{"lut":["cinematic_warm","teal_orange","bleach_bypass","..."]}},
  {"id":"bloom",            "category":"shader_pixel",  "latency_tier":"medium", "depends_on":[],          "params":{"threshold":[0.5,0.95],"intensity":[0,1]}},
  {"id":"chromatic_aberration","category":"shader_glitch","latency_tier":"light","depends_on":[],          "params":{"amount":[0,0.05]}},
  {"id":"vhs_tracking",     "category":"shader_glitch", "latency_tier":"medium", "depends_on":[],          "params":{"intensity":[0,1]}},
  {"id":"echo_trails",      "category":"time",          "latency_tier":"medium", "depends_on":[],          "params":{"frames":[2,8],"decay":[0,1]}},
  {"id":"subject_track",    "category":"ai_aware",      "latency_tier":"light",  "depends_on":["yolov8s"], "params":{"smooth":[0,1]}},
  {"id":"pose_trail",       "category":"ai_aware",      "latency_tier":"medium", "depends_on":["yolov8s_pose"],"params":{"decay":[0,1],"color":"hex"}},
  {"id":"depth_dof",        "category":"ai_aware",      "latency_tier":"medium", "depends_on":["fast_depth"], "params":{"focus_plane":[0,1],"strength":[0,1]}},
  {"id":"semantic_bg_blur", "category":"ai_aware",      "latency_tier":"medium", "depends_on":["yolov8s_seg"],"params":{"blur":[0,1]}},
  {"id":"oil_painting",     "category":"stylization",   "latency_tier":"heavy",  "depends_on":[],          "params":{"size":[3,9]}},
  ...  (~60 项，FX_REGISTRY 完整序列化)
]

## 用户风格倾向
"energetic"

## 任务
为整首曲目输出 ChoreoPlan，重点考虑：
- 段落能量与 fx 强度匹配
- 转场点（段落边界）的 transition_fx
- continuous 类 fx 用整段时间，trigger=beat/downbeat/onset 类只标 trigger 字段
```

#### 5.3.4 ChoreoPlan JSON Schema（输出契约）

```json
{
  "version": "1.0",
  "music_id": "abc123",
  "style": "energetic",
  "director_notes": "整首走 4-on-the-floor，副歌叠加视觉爆发，bridge 留白做反差",
  "segments": [
    {
      "name": "intro",
      "t_start": 0.0,
      "t_end": 18.4,
      "fx": [
        { "id": "brightness_curve", "trigger": "rms",  "params": {"k": 0.3} },
        { "id": "pan",              "trigger": "beat", "params": {"amp": 0.03} }
      ],
      "transition_out": { "id": "fade", "duration": 0.4 }
    },
    {
      "name": "chorus1",
      "t_start": 51.0,
      "t_end": 84.5,
      "fx": [
        { "id": "zoom_pulse",        "trigger": "downbeat",   "params": {"amp": 0.12} },
        { "id": "saturation_flash",  "trigger": "downbeat",   "params": {"amp": 0.6}  },
        { "id": "ai_subject_track",  "trigger": "continuous", "params": {"smooth": 0.7} },
        { "id": "shake",             "trigger": "onset",      "params": {"px": 5} }
      ],
      "transition_out": { "id": "glitch_cut", "duration": 0.15 }
    }
  ]
}
```

校验由 `pydantic` 模型完成，失败立即 fallback。

#### 5.3.5 Plan → Track 展开

LLM 不直接出每个 beat 的指令（token 浪费），由本地 `expand_plan_to_track(plan, feat)` 把段落策略展开成原 ChoreoTrack：

```python
def expand_plan_to_track(plan, feat):
    track = []
    for seg in plan["segments"]:
        for fx in seg["fx"]:
            if fx["trigger"] == "continuous":
                track.append({"t0": seg["t_start"], "t1": seg["t_end"],
                              "fx": fx["id"], "params": fx["params"]})
            elif fx["trigger"] in ("beat", "downbeat", "onset"):
                events = events_in_range(feat[fx["trigger"]+"s"],
                                         seg["t_start"], seg["t_end"])
                for t in events:
                    track.append({"t": t, "fx": fx["id"], "params": fx["params"]})
            elif fx["trigger"] == "rms":
                track.append({"t0": seg["t_start"], "t1": seg["t_end"],
                              "fx": fx["id"], "curve": rms_in_range(...)})
        if seg.get("transition_out"):
            track.append({"t": seg["t_end"], "fx": seg["transition_out"]["id"],
                          "duration": seg["transition_out"]["duration"]})
    return sorted(track, key=lambda x: x.get("t", x.get("t0", 0)))
```

#### 5.3.6 缓存 / 版本 / 失败策略

| 关注点 | 策略 |
| --- | --- |
| **幂等缓存** | key = sha1(`music_id` + `style` + `fx_catalog_version` + `model_id`)，命中直接返回 cache，省 API 费用 |
| **特效库演进** | 每次新增/移除 fx 都升 `fx_catalog_version`，旧 cache 自动失效 |
| **JSON 不合规** | 第 1 次失败 → 把校验错误塞回 user prompt 让 LLM 自纠（max retries=2）→ 仍失败 → 走 §5.2 规则版 |
| **特效幻觉** | schema 校验阶段过滤未注册 fx id；params 用 clamp 强制裁到合法范围 |
| **API 不可达** | 直接走 §5.2 规则版；前端 choreo.source 字段标记 `rule \| ai \| manual` |
| **可编辑** | 用户可在前端把 `source=ai` 改成 `manual`，PUT `/api/music/{id}/choreo` 覆盖 |

#### 5.3.7 成本估算

- 单次调用：input ≈ 2k tokens（音频特征 + FX 目录），output ≈ 1.5k tokens（ChoreoPlan）
- Claude Sonnet 4.6 ≈ $0.012/首；Opus 4.7 ≈ $0.06/首
- 命中缓存 → 0
- 即使每天 1000 首歌也仅 ~$12（Sonnet），可控

### 5.4 实时模式（可选）

需要"用户随便放外部声音也能跟节奏"时，用 `aubio.tempo`/`aubio.onset` 在线检测，延迟约 50–100ms，特效用更短的反应曲线即可。**实时模式下 LLM Choreographer 不参与**（离线规划假设不成立），退回到 §5.2 规则版。

---

## 5A. 视觉引擎（RDK X5 BPU Vision Engine）

> 这是本设计相比"普通开发板"的最大价值点。RDK X5 BPU 把"画面理解"从云端拉回设备本地，纯软件就能做出"AI 跟拍 + 人像特效 + 舞步联动"。

### 5A.1 模型选型

| 任务 | 模型 (.bin) | 输入 | 在 RDK X5 上 FPS（实测参考） | 用途 |
| --- | --- | --- | --- | --- |
| 人/物体检测 | `yolov8s` 或 `yolov5s` | 640×640 | 30–60+ | 主体跟拍、bbox 律动 |
| 人体姿态 | `yolov8s_pose` | 640×640 | 25–35 | 关键点拖尾、舞步触发 |
| 人像分割 | `yolov8s_seg` 或 `yolact` | 512×512 | 20–30 | 抠像、背景替换、人像滤镜 |
| 人脸检测（轻） | `scrfd_2.5g` | 640×640 | 60+ | 自动构图、聚焦主角 |
| 深度估计（可选） | `fast_depth` / `midas_small` | 256×256 | 30+ | 景深虚化、3D 视差 |

> 模型直接来自 **D-Robotics Model Zoo** 预编译 .bin，无需自己训练/转换；按需要二选一并行（如 检测 + 姿态）即可吃满 10 TOPS。

### 5A.2 推理管线

```
v4l2/hobot_sensor (CSI) ──▶  zero-copy NV12  ──▶  GStreamer
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       hobotfilter     hobotfilter      tee → 编码主路
                       (yolov8 det)    (pose)
                              │               │
                              ▼               ▼
                          dets[]          keypoints[]
                              └──────┬────────┘
                                     ▼
                          VisionState (asyncio)
                          { subjects:[{bbox, id, kpts}], primary_id }
```

- 用 **GStreamer + `hobot_codec`** 让推理与采集零拷贝同管线，CPU 几乎不占。
- 多模型并行：RDK X5 BPU 支持网络切换（context switch）或多网络共驻；YOLO + Pose 同帧推理在 X5 上 30fps 可达。
- 跟踪：在检测后接 `ByteTrack`（CPU 极轻量）保持 ID 稳定，避免特效在主体间乱跳。

### 5A.3 视觉事件

发布到 EventBus 的标准事件：

```python
{ "type": "vision.subject", "t": 12.34,
  "primary": { "id": 7, "bbox": [x,y,w,h], "score": 0.92 },
  "all": [ ... ] }

{ "type": "vision.pose", "t": 12.34,
  "id": 7, "kpts": [[x,y,score], ... 17 个 COCO 关键点] }

{ "type": "vision.motion", "t": 12.34,
  "id": 7, "intensity": 0.73 }   # 关键点速度均值，越大越"嗨"
```

### 5A.4 节奏 × 视觉融合规则（示例）

| 触发 | 行为 |
| --- | --- |
| 主拍 + 主体存在 | 数字 pan/zoom 让 `primary.bbox` 中心向画面中心收 30% |
| 重拍 + 多主体 | 每 4 拍切换 `primary_id`，配 fade 转场 |
| onset + pose 已知 | 在双手关键点位置画拖尾粒子，颜色随 onset 强度变化 |
| RMS 高 + motion.intensity 高 | "副歌爆发"组合：饱和爆闪 + glitch + 人像滤镜全开 |
| RMS 低 + 静止 | 切换"安静模式"：人像保留色彩，背景去饱和 + 慢速景深 |

### 5A.5 性能预算（BPU + 主流程同时跑）

| 资源 | 占比目标 |
| --- | --- |
| RDK X5 BPU | 60–80%（YOLO + Pose 双模型） |
| X5 CPU（8×Cortex-A55） | < 50%（编码 + WebRTC + FX 合成） |
| GPU/Mali | 编码独占 |
| 内存 | < 2GB |

> 若同时还要分割（segmentation），建议把分割降到 15fps 异步跑，主路径只在最近一帧 mask 上做 EMA 平滑，避免拖累 30fps 主流程。

---

## 6. 媒体管线设计

### 6.1 帧流

```
Camera (hobot_sensor/v4l2)
   │  原始帧 1280×960 NV12  @ 30fps
   ▼
Frame Source (asyncio queue, 双缓冲)
   │
   ▼
FX Compositor                    ← 接收 fx.cmd 实时调整参数
   │  支持 CPU(OpenCV) / GPU(GL/EGL)
   ▼
Encoder                          ← RDK X5 硬件 H.264（hobot_codec / v4l2 m2m）
   │
   ▼
aiortc PeerConnection            ← WebRTC 推给浏览器
```

### 6.2 时钟同步（关键）

- 浏览器播放音频，Pi 推视频；**音频在客户端权威**。
- 客户端把 `audio.currentTime` 通过 WS 周期性回报（30Hz）。
- Pi 维护 `audio_clock = currentTime + drift_compensation`，特效按 `audio_clock` 触发。
- 这样能避免"音先画后"的尴尬。

### 6.3 推荐技术栈

| 选型 | 备注 |
| --- | --- |
| MIPI Camera + hobot_sensor / v4l2 | CSI 摄像头零拷贝最低延迟 |
| **GStreamer + hobot_codec** | **采集 → 推理 → 编码 一条 pipeline 零拷贝，强烈推荐主路径** |
| **horizon_dnn (Python/C 绑定)** | 直接调 BPU 推理，灵活做 ByteTrack 等后处理 |
| OpenCV (cv2) | remap/HSV/裁剪/绘制 bbox/keypts |
| Pillow / numpy | 文字/叠加层 |
| aiortc | Python 原生 WebRTC，对接前端容易 |
| FastAPI + Uvicorn | API/WS 主框架 |

---

## 7. 接口设计

### 7.1 REST

```
POST  /api/music/upload                  multipart → { music_id }
POST  /api/music/{id}/analyze            异步触发 BeatEngine → { task_id }
GET   /api/music/{id}                    BPM、beats 数、时长、波形预览、segments
POST  /api/music/{id}/choreo/ai          触发 LLM Choreographer 生成 ChoreoPlan
                                         body: { style?: "energetic|cinematic|minimal|chaotic",
                                                 model?: "claude-opus-4-7|claude-sonnet-4-6|gpt-...",
                                                 force_refresh?: bool }
                                         → { task_id }
GET   /api/music/{id}/choreo             当前编排
                                         → { source: "ai|rule|manual",
                                             plan: ChoreoPlan,
                                             track: ChoreoTrack,
                                             director_notes?: str,
                                             llm: { model, cached, tokens_in, tokens_out } }
PUT   /api/music/{id}/choreo             上传自定义编排（覆盖为 source=manual）
POST  /api/music/{id}/choreo/regenerate  重新跑 LLM（带新 style 提示）
POST  /api/session/start                 body: { music_id, mode: rhythm|manual|auto }
POST  /api/session/stop
GET   /api/health                        → { bpu, camera, models, llm_provider, llm_reachable }
```

### 7.2 WebSocket `/ws/control`

下行（Pi→Client）：

```json
{ "type": "status", "fps": 29.7, "lat_ms": 180, "audio_clock": 12.345,
  "bpu": { "yolo_fps": 32.1, "pose_fps": 28.4, "util": 0.71 } }
{ "type": "beat",   "t": 12.34, "kind": "down" }
{ "type": "vision", "t": 12.34, "subjects": 2, "primary_id": 7 }
```

上行（Client→Pi）：

```json
{ "type": "audio_clock", "t": 12.345 }
{ "type": "fx_override", "fx": "shake", "amp": 0.0 }
{ "type": "manual", "cmd": "snapshot" }
```

### 7.3 WebRTC

`POST /rtc/offer` 标准 SDP offer/answer，单向视频 + 可选反向 datachannel。

---

## 8. 性能与时延预算

| 路径 | 目标 | 备注 |
| --- | --- | --- |
| 摄像头 → 编码 → 浏览器 | < 250 ms | RDK X5 硬编 + WebRTC |
| 节拍触发 → 画面响应 | < 1 帧 (~33ms) | 编排预生成，运行时只查表 |
| 视觉推理 (YOLO+Pose) | < 50ms / 帧 | RDK X5 BPU 实测 |
| 视觉事件 → FX 反应 | < 1 帧 | EventBus 同进程 |
| X5 CPU（特效 + 编码 + WS） | < 50% (8 核) | 推理已下沉到 BPU |
| RDK X5 BPU 占用 | 60–80% | YOLO + Pose 并行 |
| 上传 5MB MP3 → 分析完成 | < 5 s | librosa 离线 |

> 性能不够时降级路径：①分割降到 15fps；②输出降至 960×540；③复杂特效（万花筒、glitch）按段落只在副歌启用；④YOLO 切到 `yolov5s` 更轻量。

---

## 9. 前端最小形态

- **首页**：拖拽上传音乐 → 显示波形与节拍标记。
- **编排页**：时间轴可视化（beat / downbeat / onset / RMS 四条），允许在每条上加/删特效。
- **演出页**：单一大视频窗口（WebRTC）+ 播放控制条 + 实时 FPS/延迟。
- **手动模式**：手动按按钮触发某个特效，便于调试与现场。

技术：Vite + React + WaveSurfer.js + WebRTC API。

---

## 10. 部署

```bash
# === 1) 基础系统（RDK X5 已预装 Ubuntu 22.04） ===
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv ffmpeg v4l-utils \
  libsrtp2-dev libavdevice-dev pkg-config \
  gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly}

# === 2) BPU 驱动与 SDK 安装（RDK X5 官方镜像已内置，可跳过） ===
# 若需要手动安装：
sudo apt install -y hobot-dnn hobot-codec hobot-camera
# 验证 BPU 可用
hrt_model_exec model --model_file=/opt/hobot/model/basic/yolov5s.bin --frame_count=1
# 跑官方 demo 验证视觉链路
cd /opt/hobot/demo && python3 detect.py    # 或参考 D-Robotics RDK X5 示例

# === 3) 项目 ===
git clone <repo> /opt/picoclaw && cd /opt/picoclaw
python3 -m venv --system-site-packages .venv      # system-site 让 hobot_dnn/v4l2 可见
source .venv/bin/activate
pip install -r requirements.txt

# === 4) 模型放置 ===
mkdir -p models
cp /opt/hobot/model/basic/yolov8s.bin       models/    # 检测，常驻
cp /opt/hobot/model/basic/yolov8s_pose.bin  models/    # 姿态，常驻
cp /opt/hobot/model/basic/yolov8s_seg.bin   models/    # 分割，副歌按需切入
cp /opt/hobot/model/basic/fast_depth.bin    models/    # 深度，桥段按需
cp /opt/hobot/model/basic/scrfd_2.5g.bin    models/    # 人脸，卡拉 OK 段按需
# 或从 D-Robotics Model Zoo 下载新版本

# === 5b) 资源素材：3D LUT + 光效素材 ===
mkdir -p app/fx/lut3d app/fx/shaders/{pixel,geometry,color,glitch} assets/light_leaks
# 放置 .cube 文件（cinematic_warm / teal_orange / bleach_bypass / vintage_film 等）
# 光效素材：LightLeak PNG sequence（24fps，预渲染）放到 assets/light_leaks/

# === 6) LLM Provider 配置 ===
# /etc/picoclaw/picoclaw.env （被 systemd EnvironmentFile 加载）
#   PICOCLAW_LLM=claude
#   ANTHROPIC_API_KEY=sk-ant-...
#   PICOCLAW_LLM_MODEL=claude-sonnet-4-6   # 默认；可换 opus-4-7
#   PICOCLAW_LLM_CACHE_DIR=/var/lib/picoclaw/llm_cache
sudo install -d -m 750 /var/lib/picoclaw/llm_cache

# === 7) systemd ===
sudo cp deploy/picoclaw.service /etc/systemd/system/
sudo systemctl enable --now picoclaw

# === 8) 自检 ===
curl http://<rdkx5>:8000/api/health
# 应返回 { "bpu": "ok", "camera": "ok",
#          "models": ["yolov8s","yolov8s_pose"],
#          "llm_provider": "claude", "llm_reachable": true }
```

`requirements.txt` 关键项：

```
fastapi
uvicorn[standard]
aiortc
librosa
numpy
opencv-python-headless
hobot-dnn          # 通过 system-site-packages 复用系统包（RDK X5 BPU Python 绑定）
python-multipart
soundfile
pydantic           # ChoreoPlan schema 校验
anthropic          # 主选 LLM Provider
openai             # 备选 LLM Provider（可注释掉）
tenacity           # LLM 调用重试
diskcache          # ChoreoPlan 本地缓存
moderngl           # GLSL shader 运行时（RDK X5 Mali GPU OpenGL ES）
moderngl-window    # 仅开发期预览 shader（生产可用 gst-glshader 替代）
PyOpenGL           # gst-glshader fallback / shader 测试
```

> **Shader 集成方案**：生产路径用 GStreamer `glshader` element，零拷贝串在 hobotfilter 之后；
> 开发期用 `moderngl` 单独跑 shader 单测，方便快速迭代 GLSL 片段。

---

## 11. 目录结构建议

```
picoclaw/
├── app/
│   ├── api/             FastAPI 路由
│   ├── beat/            BeatEngine（含 segments）+ rule fallback
│   │   ├── analyzer.py
│   │   ├── segments.py
│   │   └── rule_choreo.py        # §5.2 fallback
│   ├── ai_choreographer/ LLM 编排 Agent（§5.3，本次新增）
│   │   ├── provider.py           # LLMProvider / Claude / OpenAI 抽象
│   │   ├── prompts.py            # system + user prompt 模板
│   │   ├── schema.py             # ChoreoPlan pydantic 模型
│   │   ├── catalog.py            # 从 fx/ 反射出 fx_catalog 给 LLM
│   │   ├── expander.py           # Plan → Track 展开
│   │   ├── cache.py              # diskcache 包装
│   │   └── service.py            # 编排入口（orchestrate(music_id, style))
│   ├── vision/          RDK X5 BPU 推理封装 / ByteTrack / 事件发布
│   │   ├── bpu_runner.py
│   │   ├── tracker.py
│   │   └── events.py
│   ├── media/           摄像头采集 / FX / 编码 / WebRTC
│   ├── fx/              各类特效模块（按延迟/类别分文件）
│   │   ├── _registry.py          # 装饰器注册 + catalog 序列化（含 latency_tier / depends_on）
│   │   ├── digital_ptz.py        # pan/tilt/zoom/shake (light)
│   │   ├── color.py              # 亮度/饱和/色相/3d_lut/duotone (light)
│   │   ├── geometry.py           # 鱼眼/万花筒/twirl/ripple (light)
│   │   ├── overlay.py            # 光圈/频谱/歌词/lens_flare/light_leak/particles
│   │   ├── time_domain.py        # echo_trails/slit_scan，含 ring buffer
│   │   ├── stylization.py        # cartoon/sketch/oil_painting (heavy, 互斥)
│   │   └── ai_aware.py           # ⭐ Hailo 驱动 fx：subject_track/pose_trail/
│   │                             #    depth_dof/pseudo_3d_parallax/semantic_bg_blur/
│   │                             #    pose_particles/multi_person_mirror/face_filter/
│   │                             #    subject_match_cut/crowd_density_react
│   │   ├── shaders/              # GLSL fragment shaders（gst-glshader 加载）
│   │   │   ├── pixel/            # bloom.frag, vignette.frag, grain.frag,
│   │   │   │                       chromatic_aberration.frag, halftone.frag, posterize.frag, ...
│   │   │   ├── geometry/         # ripple.frag, swirl.frag, tunnel.frag, whip_pan.frag, ...
│   │   │   ├── color/            # lut3d.frag, duotone.frag, curves.frag, ...
│   │   │   └── glitch/           # vhs.frag, scanlines.frag, datamosh.frag,
│   │   │                           jpeg_artifacts.frag, roll_bar.frag, rgb_shift.frag, ...
│   │   └── lut3d/                # .cube 文件（cinematic_warm.cube, teal_orange.cube,
│   │                             #             bleach_bypass.cube, ...）
│   ├── orchestrator.py  事件总线
│   └── main.py
├── models/              BPU 模型文件（.bin 格式）
│   ├── yolov8s.bin
│   └── yolov8s_pose.bin
├── web/                 前端 (Vite + React)
├── deploy/
│   └── picoclaw.service
├── tests/
└── DESIGN.md
```

---

## 12. 里程碑

| 阶段 | 交付 | 周期 |
| --- | --- | --- |
| M1 | 地瓜机器人 RDK X5 环境 ready；摄像头 → WebRTC 推流跑通 | 3–4 天 |
| M2 | 上传音乐 → 分析 → 显示 BPM/波形/节拍点 + 段落切分 | 3–4 天 |
| M3 | FX 框架 + 5 个核心特效（pan/zoom/shake/flash/brightness）+ FX_REGISTRY catalog 反射 | 1 周 |
| **M3.5** | **AI Choreographer：LLMProvider 抽象 + Claude 接入 + ChoreoPlan schema + Plan→Track 展开 + diskcache + rule fallback** | **4–5 天** |
| **M4** | **RDK X5 BPU 视觉引擎：YOLO + Pose 推理事件发布；CPU 占用验证** | **4–5 天** |
| M5 | ChoreoTrack 执行器 + 音视频时钟同步 | 4–5 天 |
| **M6** | **AI 智能特效：智能跟拍、姿态光剑、人像滤镜上线（被 LLM Plan 引用）** | **1 周** |
| M7 | 编排可视化编辑器（显示 director_notes、style 重生成、AI/manual 切换）+ 压测 + 调参 | 1 周 |

---

## 13. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
| --- | --- | --- | --- |
| BPU 驱动未正确加载 | 中 | 高 | 部署脚本检查 `dmesg \| grep bpu` 状态，未达标自动报错 |
| BPU 多模型并发资源争抢 | 中 | 中 | 使用 horizon_dnn 多网络模式；分割降到 15fps 异步 |
| 推理结果时间戳与帧不对齐 | 高 | 中 | 推理与帧绑定单调时钟 ID，FX 用最近一帧的 vision 状态 |
| X5 CPU 跑 OpenCV 不够 | 中 | 中 | 推理已下沉到 BPU；FX 优先零拷贝 + 查找表 |
| WebRTC 弱网卡顿 | 中 | 中 | 自适应码率 + TURN |
| 音视频不同步（漂移） | 高 | 中 | 客户端 `audio_clock` 周期回报 + Pi 端 PI 控制器修正 |
| 上传音频格式杂 | 中 | 低 | 服务端 ffmpeg 统一转码 wav |
| RDK X5 散热降频 | 中 | 中 | 主动散热 + 监控 `hrut_somstatus`，过热降特效 |
| **LLM API 不可达 / 超时** | **中** | **中** | **2 次重试 → 走 §5.2 规则版；前端标记 source=rule** |
| **LLM 输出非法 JSON / 引用不存在的 fx** | **中** | **中** | **pydantic schema 严格校验 + fx_id 白名单；自纠 prompt 重试 1 次后 fallback** |
| **LLM 输出特效堆叠过载，导致画面糊** | **中** | **低** | **prompt 硬约束"≤4 fx/段"；expand 后再做一次后处理裁剪** |
| **API key 泄露** | **低** | **高** | 仅放 `/etc/picoclaw/picoclaw.env` 0640 + systemd EnvironmentFile，不入库不入日志 |
| **同首歌反复触发 LLM 浪费费用** | **高** | **低** | **diskcache key=sha1(music_id+style+catalog_ver+model)，命中即返回** |
| **特效库升级使旧 Plan 引用失效** | **中** | **低** | **catalog_version 升号，旧 cache 自动失效；GET choreo 检测到失效 fx 时提示重生成** |

---

## 14. 后续可选扩展

- **LLM 视觉上下文**：开播前 Hailo 抓一帧场景描述（人数、室内/室外、光线）作为额外 context 喂 LLM，让导演知道"这是一场 3 人合舞还是单人弹唱"。
- **多 LLM 投票/对比**：同时跑 Claude + GPT，对比两个 ChoreoPlan，前端让用户挑或自动融合。
- **风格微调**：在用户调过几首歌的偏好后，用 few-shot 把"这个用户喜欢什么风格"塞进 system prompt。
- **本地小模型 fallback**：未来若有 Pi-side 小 LLM（如 Phi-4-mini），离线/无网时也能产出基础 ChoreoPlan。
- **多人互动**：用 ID 跟踪 + 姿态相似度匹配，让两位舞者的对称动作触发"镜像分屏"。
- **手势控制**：举手/比心 → 触发指定特效，无需触屏。
- **导出**：把"画面 + 音乐"用 ffmpeg 录成 mp4，一键分享。
- **3D 视差**：Hailo 跑深度估计 → 把人像分层抠出，做"伪 3D"跟拍。

---

**版本**：v0.4.1（纯软件 + 地瓜机器人 RDK X5 BPU + LLM Choreographer + 扩展 FX 库版）· 2026-05-13

**变更日志**：
- **v0.4.1 (2026-05-13)**：**平台切换 ——** 从 Raspberry Pi 5 + Hailo-8L 迁移至**地瓜机器人 RDK X5**；BPU 驱动 fx 提到第一位，每个 fx 标 latency_tier (🟢🟡🔴)，新增 8 个 BPU-aware fx（depth_dof / pseudo_3d_parallax / semantic_bg_blur / pose_particles / multi_person_mirror / face_filter / subject_match_cut / crowd_density_react）；新增 4.3 GPU Shader 类（28 个 shader fx，覆盖像素/几何/调色/glitch）；新增 4.8 LLM 选用约束；§5.3.3 prompt 注入 latency/BPU 硬约束；§10/11 更新为 RDK X5 部署流程（hobot-dnn、hobot_codec、.bin 模型格式、D-Robotics Model Zoo）。
- v0.4 (2026-05-07)：新增 §5.3 AI Choreographer (LLM Agent)；§5.2 降级为 rule fallback；§5.1 增加段落自动切分；§7.1 新增 `/api/music/{id}/choreo/ai` 与 source 字段；§10/11 接入 LLM Provider 配置与目录；§12 新增 M3.5 里程碑；§13 新增 LLM 相关 6 项风险。
- v0.3 (2026-05-06)：纯软件 + Raspberry Pi 5 + Hailo-8L 版本初稿。
