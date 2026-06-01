"""
机械臂 WebSocket 控制接口

协议 (JSON):
  Client → Server:
    {"cmd": "connect"}
    {"cmd": "disconnect"}
    {"cmd": "set_degrees", "joints": {"1": 180.0, ...}}   # servo_id(str) -> degrees
    {"cmd": "get_degrees"}

  Server → Client:
    {"type": "connected",    "joints": {"1": 180.0, ...}}
    {"type": "disconnected"}
    {"type": "degrees",      "joints": {"1": 180.0, ...}}
    {"type": "error",        "message": "..."}
    {"type": "status",       "connected": bool}
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
from fastapi import WebSocket, WebSocketDisconnect

# 把 bambot_sdk 加入路径
_sdk_dir = os.path.join(os.path.dirname(__file__), "..", "..", "bambot_sdk")
if _sdk_dir not in sys.path:
    sys.path.insert(0, _sdk_dir)

try:
    from scs_servo_sdk import ScsServoSDK, position_to_degrees, degrees_to_position
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

ARM_PORT   = "/dev/ttyACM0"
ARM_BAUD   = 1_000_000
ARM_IDS    = [1, 2, 3, 4, 5, 6]

# ── 安全参数 ──────────────────────────────────────────────────────
# 加速度 0~254：值越小越柔和。50 = 正常，20 = 非常平滑
ARM_ACCEL      = 20
# 位置模式最大速度 0~4095（0 = 不限速，容易崩）
# 300 ≈ 最大速度的 7%，归位用；日常控制可适当调高
ARM_HOME_SPEED = 300
ARM_CTRL_SPEED = 600   # 键盘/滑条控制时的速度限制

# 连接后自动归位的角度（单位：度）
# 修改这里即可更换归位姿态
HOME_DEGREES: dict[int, float] = {
    1: 207.9,   # Rotation
    2: 87.9,    # Pitch
    3: 198.5,   # Elbow
    4: 250.6,   # Wrist_Pitch
    5: 178.9,   # Wrist_Roll
    6: 194.7,   # Jaw
}


class ArmManager:
    """单例，持有 SDK 连接，供 WebSocket handler 使用。"""

    def __init__(self):
        self._sdk: "ScsServoSDK | None" = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._sdk is not None

    # ── connect ──────────────────────────────────────────────────

    async def connect(self) -> dict:
        """打开串口，初始化各关节，返回当前关节角度。"""
        if not _SDK_AVAILABLE:
            raise RuntimeError("scs_servo_sdk 未安装，请确认 bambot_sdk 目录存在")

        async with self._lock:
            if self._sdk is not None:
                return await self._read_all_degrees()

            sdk = ScsServoSDK(half_duplex=False)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sdk.connect(ARM_PORT, ARM_BAUD)
            )

            # 初始化关节：位置模式 + 加速度 + 速度限制 + 开力矩
            for sid in ARM_IDS:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.set_position_mode(s)
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.write_acceleration(s, ARM_ACCEL)
                    )
                    # 用归位速度（慢速），防止上电时急冲
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.write_wheel_speed(s, ARM_HOME_SPEED)
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.write_torque_enable(s, True)
                    )
                except Exception as e:
                    print(f"[ArmManager] init joint {sid} error: {e}")

            self._sdk = sdk

            # 归位到预设姿态（此时速度已限制为 ARM_HOME_SPEED）
            home_positions = {sid: degrees_to_position(deg) for sid, deg in HOME_DEGREES.items()}
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sdk.sync_write_positions(home_positions)
                )
                print(f"[ArmManager] homing at speed={ARM_HOME_SPEED} accel={ARM_ACCEL}")
            except Exception as e:
                print(f"[ArmManager] home move error: {e}")

            # 归位完成后切换到正常控制速度
            for sid in ARM_IDS:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.write_wheel_speed(s, ARM_CTRL_SPEED)
                    )
                except Exception:
                    pass

            return {str(sid): deg for sid, deg in HOME_DEGREES.items()}

    # ── disconnect ───────────────────────────────────────────────

    async def disconnect(self):
        async with self._lock:
            if self._sdk is None:
                return
            sdk = self._sdk
            self._sdk = None
            for sid in ARM_IDS:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.write_torque_enable(s, False)
                    )
                except Exception:
                    pass
            try:
                await asyncio.get_event_loop().run_in_executor(None, sdk.disconnect)
            except Exception:
                pass

    # ── set degrees ──────────────────────────────────────────────

    async def set_degrees(self, joints: dict) -> None:
        """joints: {servo_id_str: degrees_float}"""
        if self._sdk is None:
            raise RuntimeError("未连接机械臂")
        positions = {int(k): degrees_to_position(float(v)) for k, v in joints.items()}
        sdk = self._sdk
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: sdk.sync_write_positions(positions)
        )

    # ── director 专用：带锁的读写（避免与 connect/disconnect 交错） ──

    async def sync_write_degrees(self, joints: dict) -> None:
        """joints: {servo_id(int): degrees(float)}。带锁、角度夹紧 0~360。"""
        async with self._lock:
            if self._sdk is None:
                return
            positions = {
                int(sid): degrees_to_position(max(0.0, min(360.0, float(deg))))
                for sid, deg in joints.items()
            }
            sdk = self._sdk
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sdk.sync_write_positions(positions)
            )

    async def read_degrees(self, ids: list) -> dict:
        """带锁读取指定舵机当前角度。返回 {servo_id(int): degrees|None}。"""
        async with self._lock:
            if self._sdk is None:
                return {}
            sdk = self._sdk
            out = {}
            for sid in ids:
                try:
                    pos = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sid: sdk.read_position(s)
                    )
                    out[sid] = position_to_degrees(pos)
                except Exception:
                    out[sid] = None
            return out

    # ── get degrees ──────────────────────────────────────────────

    async def get_degrees(self) -> dict:
        if self._sdk is None:
            raise RuntimeError("未连接机械臂")
        return await self._read_all_degrees()

    async def _read_all_degrees(self) -> dict:
        sdk = self._sdk
        result = {}
        for sid in ARM_IDS:
            try:
                pos = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sid: sdk.read_position(s)
                )
                result[str(sid)] = round(position_to_degrees(pos), 1)
            except Exception:
                result[str(sid)] = None
        return result


# 全局单例
_arm = ArmManager()


async def arm_ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    from app.api.arm_director import director

    # 发送初始状态
    await websocket.send_text(json.dumps({
        "type": "status",
        "connected": _arm.is_connected,
        "sdk_available": _SDK_AVAILABLE,
        "director_enabled": director.enabled,
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue

            cmd = msg.get("cmd")

            if cmd == "connect":
                try:
                    joints = await _arm.connect()
                    await websocket.send_text(json.dumps({"type": "connected", "joints": joints}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif cmd == "disconnect":
                # 断开前先关闭联动，避免导演继续写已关闭的串口
                if director.enabled:
                    await director.set_enabled(False)
                await _arm.disconnect()
                await websocket.send_text(json.dumps({"type": "disconnected"}))

            elif cmd == "set_degrees":
                # 联动开启时由导演独占串口，忽略手动写入
                if director.enabled:
                    continue
                try:
                    await _arm.set_degrees(msg.get("joints", {}))
                    # 不回包，减少延迟
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif cmd == "set_director":
                try:
                    await director.set_enabled(msg.get("enabled", False))
                    await websocket.send_text(json.dumps({
                        "type": "director", "enabled": director.enabled}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif cmd == "get_degrees":
                try:
                    joints = await _arm.get_degrees()
                    await websocket.send_text(json.dumps({"type": "degrees", "joints": joints}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"unknown cmd: {cmd}"}))

    except WebSocketDisconnect:
        pass
