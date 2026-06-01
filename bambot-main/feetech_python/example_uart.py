"""
GPIO UART 接线与使用示例
"""

# ─── 方案 A：电阻法（推荐新手） ───────────────────────────────
from scs_servo_uart import ScsServoUART, position_to_degrees
import time

sdk = ScsServoUART()
sdk.connect("/dev/serial0", baud_rate=1_000_000)

# 初始化 so-arm100 六轴机械臂
ARM_IDS = [1, 2, 3, 4, 5, 6]
for sid in ARM_IDS:
    sdk.set_position_mode(sid)
    sdk.write_torque_enable(sid, True)
    sdk.write_acceleration(sid, 50)

# 读取当前角度
positions = sdk.sync_read_positions(ARM_IDS)
for sid, raw in positions.items():
    print(f"Servo {sid}: {position_to_degrees(raw):.1f}°")

# 批量归位（所有关节 180°）
sdk.sync_write_positions_degrees({sid: 180.0 for sid in ARM_IDS})
time.sleep(2)

# 夹爪开合
sdk.write_position_degrees(6, 210.0)   # 张开
time.sleep(0.5)
sdk.write_position_degrees(6, 150.0)   # 收紧
time.sleep(0.5)

# 断开前关闭力矩
for sid in ARM_IDS:
    sdk.write_torque_enable(sid, False)
sdk.disconnect()


# ─── 方案 B：GPIO 方向控制法 ──────────────────────────────────
# from scs_servo_uart import ScsServoUART_DirectionGPIO
#
# sdk = ScsServoUART_DirectionGPIO(dir_pin=18)  # GPIO18 = 物理引脚12
# sdk.connect("/dev/serial0")
# ... 其余调用与方案 A 完全相同 ...
# sdk.disconnect()


# ─── 底盘控制示例（bambot-b0-base） ───────────────────────────
# WHEEL_IDS = [13, 14, 15]
# for sid in WHEEL_IDS:
#     sdk.set_wheel_mode(sid)
#
# def move_forward(speed=300, duration=1.0):
#     sdk.sync_write_wheel_speed({13: speed, 14: -speed, 15: speed})
#     time.sleep(duration)
#     sdk.sync_write_wheel_speed({13: 0, 14: 0, 15: 0})
#
# def turn_left(speed=300, duration=0.5):
#     sdk.sync_write_wheel_speed({13: -speed, 14: 0, 15: speed})
#     time.sleep(duration)
#     sdk.sync_write_wheel_speed({13: 0, 14: 0, 15: 0})
#
# move_forward(speed=300, duration=1.0)
# turn_left()
