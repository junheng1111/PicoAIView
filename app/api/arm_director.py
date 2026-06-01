"""
机械臂导演（ArmDirector）

把后端视觉/运镜信号转成机械臂动作：
- servo1(底座旋转) + servo2(俯仰)：人居中跟踪（超阈值才纠正一下的离散 nudge）
- servo3(肘) / servo4(腕俯仰) / servo5(腕滚转)：按「当前激活的 FX（运镜模式）」做轨迹表演
- servo6(夹爪)：不参与

所有运动以 HOME_DEGREES 为基准；切换运镜模式前先确认 servo3-5 回到 home。
作为 asyncio 任务运行，复用 ArmManager 的带锁读写，避免裸线程操作共享串口。

安全：所有增益/幅度/范围都是模块顶部常量，默认保守。
      ⚠ 居中方向符号 CENTER_GAIN_X/Y 取决于机械臂物理安装，首次上机若方向反了请翻转符号。
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Dict, Optional

from app.api import arm_ws  # 运行时通过 arm_ws._arm / arm_ws.HOME_DEGREES 访问

# ── 关节角色 ──────────────────────────────────────────────────────
CENTER_JOINTS = [1, 2]        # 人居中
TRAJ_JOINTS   = [3, 4, 5]     # 运镜轨迹表演

# ── 人居中参数 ────────────────────────────────────────────────────
CENTER_TH      = 0.15    # 偏移死区：中心偏离画面中心超过 15% 才纠正
CENTER_GAIN_X  = 30.0    # 水平：每单位归一化偏移对应的底座角度增益（含符号，⚠需上机微调）
CENTER_GAIN_Y  = 20.0    # 垂直：每单位偏移对应的俯仰角度增益（含符号，⚠需上机微调）
CENTER_RANGE   = 25.0    # 居中关节偏离 home 的最大角度
CENTER_STEP    = 3.0     # 单帧最大纠正步长（度），实现「移动一下」而非瞬移

# ── 运镜轨迹参数 ──────────────────────────────────────────────────
TRAJ_MAX_AMP   = 15.0    # 轨迹关节相对 home 的最大摆幅（度）
TRAJ_STEP      = 4.0     # 单帧最大变化（度），防止幅度/模式切换跳变
HOME_TOL       = 8.0     # 模式切换时判定「已在 home」的容差（度）
HOME_SETTLE    = 0.4     # 写 home 后等待到位的时间（秒）

LOOP_HZ        = 10.0
LOOP_DT        = 1.0 / LOOP_HZ

# ── FX(运镜模式) → 轨迹映射 ───────────────────────────────────────
# 每项: (servo_id, 频率Hz, 相对幅度0~1)
# 轨迹 = home + TRAJ_MAX_AMP * intensity * rel * sin(2π·freq·t)
_DEFAULT_TRAJ = [(3, 0.20, 0.30)]   # 极轻微呼吸
FX_TRAJECTORY: Dict[str, list] = {
    "pan":          [(5, 0.30, 1.00)],               # 腕滚转缓慢横摆
    "zoom_pulse":   [(3, 0.80, 0.80)],               # 肘随节拍前推
    "subject_zoom": [(3, 0.25, 0.50), (4, 0.25, 0.40)],
    "glitch":       [(4, 4.00, 0.50)],               # 腕俯仰小幅快抖
    "shake":        [(4, 5.00, 0.60), (5, 5.00, 0.40)],
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _slew(cur, target, step):
    """把 cur 朝 target 移动，单步不超过 step。"""
    d = target - cur
    if d > step:
        return cur + step
    if d < -step:
        return cur - step
    return target


class ArmDirector:
    def __init__(self):
        self._enabled = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # 当前已下发的目标角度（servo_id -> degrees），仅 1-5
        self._targets: Dict[int, float] = {}
        self._cur_fx: Optional[str] = None
        self._t0 = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── 外部接口 ─────────────────────────────────────────────────

    async def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        if enabled:
            if not arm_ws._arm.is_connected:
                raise RuntimeError("请先连接机械臂再开启联动")
            # 以 home 为起点
            self._targets = {sid: arm_ws.HOME_DEGREES[sid] for sid in (CENTER_JOINTS + TRAJ_JOINTS)}
            self._cur_fx = None
            self._t0 = time.monotonic()
            await arm_ws._arm.sync_write_degrees(dict(self._targets))
            self._enabled = True
            print("[ArmDirector] 联动已开启")
        else:
            self._enabled = False
            # 平滑归位（仅在仍连接时）
            try:
                if arm_ws._arm.is_connected:
                    home = {sid: arm_ws.HOME_DEGREES[sid] for sid in (CENTER_JOINTS + TRAJ_JOINTS)}
                    await arm_ws._arm.sync_write_degrees(home)
            except Exception as e:
                print(f"[ArmDirector] 归位失败: {e}")
            print("[ArmDirector] 联动已关闭")

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ── 控制循环 ─────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                if self._enabled and arm_ws._arm.is_connected:
                    await self._tick()
                else:
                    # 未启用时若误标记 enabled（断连），复位
                    if self._enabled and not arm_ws._arm.is_connected:
                        self._enabled = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ArmDirector] tick error: {e}")
            await asyncio.sleep(LOOP_DT)

    async def _tick(self) -> None:
        from app.main import get_orchestrator
        orch = get_orchestrator()
        if orch is None or orch.bpu_runner is None:
            return

        vs = orch.bpu_runner.vision_state()
        rms = orch.compositor.current_rms if orch.compositor else 0.0
        fx = orch.current_active_fx()
        intensity = _clamp(max(vs.motion_intensity if vs else 0.0, rms), 0.0, 1.0)

        # ── 模式切换：先确认轨迹关节回到 home ──
        if fx != self._cur_fx:
            await self._ensure_home(TRAJ_JOINTS)
            self._cur_fx = fx
            self._t0 = time.monotonic()

        # ── 人居中（servo1/2 离散 nudge）──
        if vs and vs.subject_count > 0:
            pb = vs.primary_bbox()
            if pb:
                cx = pb[0] + pb[2] / 2.0
                cy = pb[1] + pb[3] / 2.0
                off_x = cx - 0.5
                off_y = cy - 0.5
                if abs(off_x) > CENTER_TH:
                    self._nudge_center(1, off_x, CENTER_GAIN_X)
                if abs(off_y) > CENTER_TH:
                    self._nudge_center(2, off_y, CENTER_GAIN_Y)

        # ── 运镜轨迹（servo3/4/5）──
        t = time.monotonic() - self._t0
        traj = FX_TRAJECTORY.get(fx) or _DEFAULT_TRAJ
        # 本帧轨迹关节先默认停在 home，再被映射覆盖
        traj_target = {sid: arm_ws.HOME_DEGREES[sid] for sid in TRAJ_JOINTS}
        for sid, freq, rel in traj:
            if sid not in traj_target:
                continue
            amp = TRAJ_MAX_AMP * intensity * rel
            traj_target[sid] = arm_ws.HOME_DEGREES[sid] + amp * math.sin(2 * math.pi * freq * t)
        for sid in TRAJ_JOINTS:
            cur = self._targets.get(sid, arm_ws.HOME_DEGREES[sid])
            self._targets[sid] = _slew(cur, traj_target[sid], TRAJ_STEP)

        # ── 一次性下发 1-5 ──
        await arm_ws._arm.sync_write_degrees(dict(self._targets))

    def _nudge_center(self, sid: int, off: float, gain: float) -> None:
        home = arm_ws.HOME_DEGREES[sid]
        cur = self._targets.get(sid, home)
        step = _clamp(gain * off, -CENTER_STEP, CENTER_STEP)
        nxt = _clamp(cur + step, home - CENTER_RANGE, home + CENTER_RANGE)
        self._targets[sid] = nxt

    async def _ensure_home(self, ids: list) -> None:
        """读回各关节，偏离 home 超容差则写 home 并等待到位。"""
        try:
            cur = await arm_ws._arm.read_degrees(ids)
        except Exception:
            cur = {}
        need = {}
        for sid in ids:
            c = cur.get(sid)
            home = arm_ws.HOME_DEGREES[sid]
            if c is None or abs(c - home) > HOME_TOL:
                need[sid] = home
            self._targets[sid] = home
        if need:
            await arm_ws._arm.sync_write_degrees(need)
            await asyncio.sleep(HOME_SETTLE)


# 全局单例
director = ArmDirector()
