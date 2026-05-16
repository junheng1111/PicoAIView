# PicoView

** 实时音乐视觉编排系统**

摄像头画面跟随音乐节奏产生视觉律动。上传一段音乐，AI 导演自动分析节奏、编排镜头特效；BPU 实时识别画面中的人体，让特效"看懂"并跟随主体。

---

## 系统架构

```
浏览器 Web UI
    │ HTTP REST / WebSocket / MJPEG
    ▼
FastAPI 服务（RDK X5，port 8000）
    ├── BeatEngine      librosa 离线分析：BPM / beats / downbeats / onsets / RMS / 段落
    ├── AI Choreographer  OpenClaw 按音频特征生成 ChoreoPlan
    ├── FX Compositor   4层叠加：主题滤镜 / 连续FX / 瞬态FX / 常驻层
    └── Vision Engine  ，YOLOv8 人体检测 + 姿态估计
```

---

## 硬件要求

| 组件 | 说明 |
|---|---|
| MIPI CSI 摄像头 | RDK X5 官方摄像头或兼容模组 |
| Ubuntu 22.04 | RDK X5 官方镜像，预装 hobot-dnn |

NPU 不可用时系统自动降级：改用 OpenCV 软件推理继续运行。

---

## 安装

```bash
# 创建虚拟环境（需继承系统 site-packages 以访问 hobot_dnn）
python3 -m venv venv --system-site-packages
source venv/bin/activate

pip install -r requirements.txt
```

---

## 配置

复制示例环境变量文件并按需修改：

```bash
cp deploy/picoclaw.env.example /etc/picoclaw/picoclaw.env
```

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PICOCLAW_LLM` | `openclaw` | LLM 后端：`openclaw` / `claude` / `openai` / `minimax` |
| `OPENCLAW_GATEWAY_TOKEN` | — | OpenClaw 本地网关 token |
| `ANTHROPIC_API_KEY` | — | 使用 Claude 时填写 |
| `OPENAI_API_KEY` | — | 使用 OpenAI 时填写 |
| `MINIMAX_API_KEY` | — | 使用 MiniMax 时填写 |
| `PICOCLAW_CAM_W/H/FPS` | `1920/1080/30` | 摄像头分辨率与帧率 |

---

## 启动 / 停止

```bash
# 后台启动（日志写入 /tmp/pico_server.log）
bash open.sh

# 前台启动（直接看日志）
bash open.sh --fg

# 停止并释放 BPU / 摄像头资源
bash kill.sh
```

浏览器访问 `http://<IP>:8000`

---

## 使用流程

1. **上传音乐** — 支持 MP3 / WAV / FLAC，拖拽或点击选择
2. **分析节奏** — 点击「🔍 分析节奏」，BeatEngine 自动提取 BPM / 节拍 / 段落
3. **生成编排** — 点击「✨ 生成编排」，在风格对话框输入风格描述（可选），AI 导演根据音频特征选配特效序列
4. **开始预览** — 点击「▶ 开始预览」，选择编排后同步播放音乐，摄像头画面实时叠加特效
5. **实时调参** — 侧边栏可调节常驻效果强度、切换主题滤镜、在 FX 测试区单独激活任意特效

---

## 特效分类

| 分类 | 说明 | 代表 FX |
|---|---|---|
| 运镜 | 数字 PTZ，不动相机 | `pan` `zoom_pulse` `shake` `subject_zoom` |
| 主题滤镜 | 全帧色彩风格，持续生效 | `theme_warm` `theme_cool` `theme_neon` `theme_cinematic` `theme_pink` `theme_vintage` |
| 叠加特效 | 叠加层，支持透明混合 | `pink_halo` `flash` `vignette` `film_grain` |
| 时间域 | 基于历史帧缓冲 | `echo_trails` |
| 几何 | 画面形变 | `barrel_distort` `wave_warp` |
| 风格化 | 全帧风格迁移 | `duotone` `glitch` |
| AI 感知 | 依赖 BPU 检测结果 | `depth_dof` `silhouette` |

常驻层（始终生效，无需在编排中添加）：`subject_zoom`（主体跟焦）、`zoom_pulse`（节拍脉冲）、`brightness_curve`（RMS 呼吸亮度）

---

## AI 编排原理

```
音频特征 (BPM / beats / segments / RMS)
    + FX 目录（可用特效 + 参数范围）
    + 用户风格描述
    ──▶ OpenClaw 
    ──▶ ChoreoPlan（段落级方案）
    ──▶ expander 展开成逐拍 track
    ──▶ FxCompositor 每帧按 audio_clock 查找匹配事件渲染
```

ChoreoPlan 包含：全曲主题、段落列表（每段有起止时间、子主题、FX 列表、转场）。  
编排结果持久化到 `data/choreos/<choreo_id>.json`，同一首歌可保存多套编排。

规则 fallback：LLM 不可用时自动降级到确定性规则（beats→pan，downbeats→zoom+flash，onsets→shake）。

---

## 目录结构

```
pico_view/
├── app/
│   ├── ai_choreographer/   LLM 编排：provider / prompts / schema / expander / cache
│   ├── api/                FastAPI 路由 + WebSocket 状态推送
│   ├── beat/               librosa 音频分析 + 规则 fallback
│   ├── fx/                 特效库（color / overlay / geometry / glitch / ai_aware …）
│   ├── media/              摄像头采集 + FX 合成器
│   ├── vision/             BPU 推理封装 + 目标跟踪
│   ├── orchestrator.py     主协调器：音频时钟 / 节拍触发 / 帧渲染
│   └── main.py             FastAPI app 入口 + lifespan
├── web/                    前端（纯 HTML + CSS + JS，无构建步骤）
├── models/                 NPU .bin 模型文件
├── data/                   持久化（音乐库 / 编排文件 / beat 特征缓存）
├── deploy/                 systemd service + 环境变量示例
├── open.sh                 启动脚本
└── kill.sh                 停止并释放 NPU / 摄像头资源
```

---

## 监控日志

```bash
# 实时查看所有日志
tail -f /tmp/pico_server.log

# 只看 AI 编排相关
tail -f /tmp/pico_server.log | grep -E "AIChoreographer|OpenClawProvider"

# 只看 BPU 推理性能
tail -f /tmp/pico_server.log | grep BpuRunner
```

---

## 作为 systemd 服务运行

```bash
cp deploy/picoclaw.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable picoclaw
systemctl start picoclaw
```

---

## 开发说明

- **前端无构建步骤**：直接修改 `web/` 下的文件，刷新浏览器即可。修改 JS/CSS 后需更新 `index.html` 中的版本号（`?v=N`）避免缓存。
- **添加新特效**：在 `app/fx/` 下实现函数，用 `@register_fx(fx_id=..., category=...)` 注册，重启后自动出现在 FX 目录和 AI 编排可用列表中。
- **切换 LLM**：设置环境变量 `PICOCLAW_LLM=claude|openai|minimax|openclaw`，重启服务生效。
