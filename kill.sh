#!/bin/bash
# kill.sh — 干净地停止 PicoClaw 并释放 BPU / CSI / 相机驱动资源
#
# 用法：bash kill.sh

WAIT_TERM=6      # SIGTERM 后等待优雅退出秒数
WAIT_KILL=2      # SIGKILL 后等待秒数
WAIT_DRIVER=3    # 等待内核驱动释放句柄秒数

# ── 找进程 ───────────────────────────────────────────────────────────
find_pids() {
    {
        pgrep -f "uvicorn app.main:app" 2>/dev/null
        pgrep -f "python.*app\.main"    2>/dev/null
        pgrep -f "python.*pico_view"    2>/dev/null
    } | sort -u | grep -v "^${BASHPID}$" || true
}

PIDS=$(find_pids)

if [ -z "$PIDS" ]; then
    echo "[kill.sh] 没有找到运行中的 PicoClaw 进程"
else
    echo "[kill.sh] 找到进程: $(echo $PIDS | xargs)"

    # Step 1: SIGTERM → 触发 Python lifespan 清理（bpu.stop → del models）
    echo "[kill.sh] SIGTERM → 等待优雅退出（最多 ${WAIT_TERM}s）..."
    kill -TERM $PIDS 2>/dev/null || true

    for i in $(seq 1 $WAIT_TERM); do
        sleep 1
        [ -z "$(find_pids)" ] && { echo "[kill.sh] 进程已正常退出（${i}s）"; break; }
    done

    # Step 2: SIGKILL fallback
    REMAINING=$(find_pids)
    if [ -n "$REMAINING" ]; then
        echo "[kill.sh] 仍有进程存活，强制 SIGKILL: $(echo $REMAINING | xargs)"
        kill -KILL $REMAINING 2>/dev/null || true
        sleep $WAIT_KILL
    fi
fi

# ── Step 3: BPU 驱动层清理 ───────────────────────────────────────────
echo "[kill.sh] 清理 BPU 驱动层资源..."

# 找到所有可能的 BPU 相关设备节点
BPU_DEVS=""
for dev in /dev/bpu0 /dev/bpu1 /dev/ion /dev/hbmem /dev/hbmem0 /dev/hbmem1; do
    [ -e "$dev" ] && BPU_DEVS="$BPU_DEVS $dev"
done

# 杀掉仍持有 BPU 设备的残余进程
if [ -n "$BPU_DEVS" ]; then
    for dev in $BPU_DEVS; do
        USERS=$(fuser "$dev" 2>/dev/null || true)
        if [ -n "$USERS" ]; then
            echo "[kill.sh] $dev 仍被 PID $USERS 持有，强制清理..."
            kill -KILL $USERS 2>/dev/null || true
        fi
    done
fi

# ── Step 3b: 相机驱动层清理（hobot_vio CSI + v4l2）────────────────────
echo "[kill.sh] 清理相机驱动层资源..."

# RDK X5 相机相关设备节点：CSI/VSE/ISP + v4l2
CAM_DEVS=""
for dev in \
    /dev/cam0 /dev/cam1 /dev/cam2 /dev/cam3 \
    /dev/vse0 /dev/vse1 \
    /dev/isp0 /dev/isp1 \
    /dev/video0 /dev/video1 /dev/video2 /dev/video3 \
    /dev/video4 /dev/video5 /dev/video6 /dev/video7; do
    [ -e "$dev" ] && CAM_DEVS="$CAM_DEVS $dev"
done

if [ -n "$CAM_DEVS" ]; then
    for dev in $CAM_DEVS; do
        USERS=$(fuser "$dev" 2>/dev/null || true)
        if [ -n "$USERS" ]; then
            echo "[kill.sh] $dev 仍被 PID $USERS 持有，强制清理..."
            kill -KILL $USERS 2>/dev/null || true
        fi
    done
fi

# 尝试通过 sysfs 重置 BPU（RDK X5 支持时有效）
for reset_path in \
    /sys/class/bpu/bpu0/reboot \
    /sys/devices/platform/bpu/reset \
    /sys/kernel/debug/bpu/reset; do
    if [ -w "$reset_path" ]; then
        echo "[kill.sh] sysfs BPU reset: $reset_path"
        echo 1 > "$reset_path" 2>/dev/null || true
        break
    fi
done

echo "[kill.sh] 等待驱动释放 ${WAIT_DRIVER}s..."
sleep $WAIT_DRIVER

# ── Step 4: 诊断 ────────────────────────────────────────────────────
BPU_BUSY=""
if [ -n "$BPU_DEVS" ]; then
    for dev in $BPU_DEVS; do
        USERS=$(fuser "$dev" 2>/dev/null || true)
        [ -n "$USERS" ] && BPU_BUSY="$BPU_BUSY $dev(pid:$USERS)"
    done
fi

CAM_BUSY=""
if [ -n "$CAM_DEVS" ]; then
    for dev in $CAM_DEVS; do
        USERS=$(fuser "$dev" 2>/dev/null || true)
        [ -n "$USERS" ] && CAM_BUSY="$CAM_BUSY $dev(pid:$USERS)"
    done
fi

if [ -n "$BPU_BUSY" ]; then
    echo "[kill.sh] ⚠  BPU 设备仍被占用: $BPU_BUSY"
    echo "[kill.sh]    下次启动若超过 12s 未加载模型，服务会自动降级（无 BPU 推理）。"
    echo "[kill.sh]    彻底释放请执行: reboot"
else
    echo "[kill.sh] ✓  BPU 设备空闲，可以安全重启"
fi

if [ -n "$CAM_BUSY" ]; then
    echo "[kill.sh] ⚠  相机设备仍被占用: $CAM_BUSY"
    echo "[kill.sh]    下次启动 open_cam() 可能需要最多 25s 重试才能打开相机。"
    echo "[kill.sh]    彻底释放请执行: reboot"
else
    echo "[kill.sh] ✓  相机设备空闲，可以安全重启"
fi

echo "[kill.sh] 完成"
