"""
使用示例 — BamBot / so-arm100 机械臂

依赖安装:
    pip install pyserial
"""

from scs_servo_sdk import ScsServoSDK, scan_servos
import time

# ─── 1. 扫描在线舵机 ───────────────────────────────────────────
# online_ids = scan_servos("COM3")  # Windows
# online_ids = scan_servos("/dev/ttyUSB0")  # Linux

# ─── 2. 连接与初始化 ───────────────────────────────────────────
sdk = ScsServoSDK()
sdk.connect("COM3")  # 改成你的串口号

# so-arm100 有 6 个旋转关节，舵机 ID 1~6
ARM_IDS = [1, 2, 3, 4, 5, 6]

for sid in ARM_IDS:
    sdk.set_position_mode(sid)       # 切换为位置模式
    sdk.write_torque_enable(sid, True)  # 开启力矩
    sdk.write_acceleration(sid, 50)     # 设置加速度

# ─── 3. 读取当前位置 ───────────────────────────────────────────
print("当前各关节角度:")
positions = sdk.sync_read_positions(ARM_IDS)
for sid, pos in positions.items():
    from scs_servo_sdk import position_to_degrees
    print(f"  Servo {sid}: raw={pos}, degrees={position_to_degrees(pos):.1f}°")

# ─── 4. 单关节角度控制 ─────────────────────────────────────────
sdk.write_position_degrees(1, 180.0)  # Rotation → 180°
time.sleep(1)

sdk.write_position_degrees(2, 90.0)   # Pitch → 90°
time.sleep(1)

# ─── 5. 批量角度控制（同步写，一次串口包） ───────────────────
sdk.sync_write_positions_degrees({
    1: 180.0,   # Rotation
    2: 180.0,   # Pitch
    3: 180.0,   # Elbow
    4: 180.0,   # Wrist_Pitch
    5: 180.0,   # Wrist_Roll
    6: 200.0,   # Jaw（张开）
})
time.sleep(2)

# ─── 6. 夹爪控制 ──────────────────────────────────────────────
sdk.write_position_degrees(6, 200.0)  # 张开
time.sleep(0.5)
sdk.write_position_degrees(6, 150.0)  # 收紧
time.sleep(0.5)

# ─── 7. 底盘车轮控制（bambot-b0-base，舵机 ID 13/14/15） ──────
WHEEL_IDS = [13, 14, 15]

for sid in WHEEL_IDS:
    sdk.set_wheel_mode(sid)

def move_forward(speed: int = 300, duration: float = 1.0):
    """前进"""
    sdk.sync_write_wheel_speed({13: speed, 14: -speed, 15: speed})
    time.sleep(duration)
    sdk.sync_write_wheel_speed({13: 0, 14: 0, 15: 0})

def move_backward(speed: int = 300, duration: float = 1.0):
    """后退"""
    sdk.sync_write_wheel_speed({13: -speed, 14: speed, 15: -speed})
    time.sleep(duration)
    sdk.sync_write_wheel_speed({13: 0, 14: 0, 15: 0})

def turn_left(speed: int = 300, duration: float = 0.5):
    """左转"""
    sdk.sync_write_wheel_speed({13: -speed, 14: 0, 15: speed})
    time.sleep(duration)
    sdk.sync_write_wheel_speed({13: 0, 14: 0, 15: 0})

def turn_right(speed: int = 300, duration: float = 0.5):
    """右转"""
    sdk.sync_write_wheel_speed({13: speed, 14: 0, 15: -speed})
    time.sleep(duration)
    sdk.sync_write_wheel_speed({13: 0, 14: 0, 15: 0})

# move_forward()
# turn_left()

# ─── 8. 断开连接 ──────────────────────────────────────────────
# 断开前停止所有车轮并关闭力矩
for sid in WHEEL_IDS:
    sdk.write_wheel_speed(sid, 0)
for sid in ARM_IDS:
    sdk.write_torque_enable(sid, False)

sdk.disconnect()
