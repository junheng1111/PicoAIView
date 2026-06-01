#!/usr/bin/env python3
"""
机械臂 UART 连通性测试脚本
- 自动检测可用串口
- 扫描在线舵机 (ID 1~6)
- 读取当前关节角度
- 执行一个小幅度来回运动测试 (默认仅测试 ID=1)

用法:
    python test_arm_uart.py                      # 使用默认端口 /dev/ttyS1
    python test_arm_uart.py --port /dev/ttyUSB0  # 指定端口
    python test_arm_uart.py --scan-only           # 只扫描，不运动
"""

import sys
import time
import argparse

# ── 添加 SDK 路径 ─────────────────────────────────────────────
sys.path.insert(0, "bambot_sdk")

try:
    from scs_servo_sdk import ScsServoSDK, scan_servos, position_to_degrees
except ImportError as e:
    print(f"[ERROR] 无法导入 scs_servo_sdk: {e}")
    sys.exit(1)

ARM_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_PORT = "/dev/ttyACM0"  # CH343 USB 控制板（内置半双工，half_duplex=False）
DEFAULT_BAUD = 1_000_000


def list_serial_ports() -> list[str]:
    """列出系统中所有可用串口"""
    import glob
    ports = (
        glob.glob("/dev/ttyS[0-9]*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
        + glob.glob("/dev/ttyAMA*")
    )
    return sorted(ports)


def test_connection(port: str, baud: int) -> bool:
    """测试串口能否打开"""
    try:
        import serial
        s = serial.Serial(port, baud, timeout=0.3)
        s.close()
        print(f"[OK]   串口 {port} 可以打开 (baud={baud})")
        return True
    except Exception as e:
        print(f"[FAIL] 串口 {port} 无法打开: {e}")
        return False


def do_scan(port: str, baud: int) -> list[int]:
    """扫描 ID 1~6 的舵机"""
    print(f"\n── 扫描舵机 (ID 1~6) on {port} @ {baud} ──")
    sdk = ScsServoSDK(half_duplex=False)
    sdk.connect(port, baud)
    found = []
    for sid in ARM_IDS:
        ok = sdk.ping(sid)
        status = "在线 ✓" if ok else "离线"
        print(f"  ID {sid}: {status}")
        if ok:
            found.append(sid)
    sdk.disconnect()
    return found


def do_read_positions(port: str, baud: int, ids: list[int]):
    """读取各关节当前角度"""
    print(f"\n── 读取当前关节角度 ──")
    sdk = ScsServoSDK(half_duplex=False)
    sdk.connect(port, baud)
    for sid in ids:
        try:
            pos = sdk.read_position(sid)
            deg = position_to_degrees(pos)
            print(f"  ID {sid}: raw={pos:4d}  degrees={deg:6.1f}°")
        except Exception as e:
            print(f"  ID {sid}: 读取失败 → {e}")
    sdk.disconnect()


def do_move_test(port: str, baud: int, test_id: int = 1):
    """
    对单个关节做小幅来回运动测试
    - 先读当前角度
    - 向 +10° 移动
    - 等 1.5s
    - 返回原始角度
    """
    print(f"\n── 运动测试 (ID={test_id}) ──")
    sdk = ScsServoSDK(half_duplex=False)
    sdk.connect(port, baud)

    try:
        # 读初始角度
        cur_pos = sdk.read_position(test_id)
        cur_deg = position_to_degrees(cur_pos)
        print(f"  当前角度: {cur_deg:.1f}°  (raw={cur_pos})")

        # 目标角度：+10°，夹在 10~350° 安全范围内
        target_deg = cur_deg + 10.0
        if target_deg > 350.0:
            target_deg = cur_deg - 10.0
        target_deg = max(10.0, min(350.0, target_deg))

        # 启用力矩、设置加速度
        sdk.write_torque_enable(test_id, True)
        sdk.write_acceleration(test_id, 30)

        print(f"  移动到: {target_deg:.1f}°  ...")
        sdk.write_position_degrees(test_id, target_deg)
        time.sleep(1.5)

        # 读取实际到达位置
        arrived_pos = sdk.read_position(test_id)
        arrived_deg = position_to_degrees(arrived_pos)
        print(f"  到达角度: {arrived_deg:.1f}°  (raw={arrived_pos})")

        # 返回原始角度
        print(f"  返回: {cur_deg:.1f}°  ...")
        sdk.write_position_degrees(test_id, cur_deg)
        time.sleep(1.5)

        final_pos = sdk.read_position(test_id)
        final_deg = position_to_degrees(final_pos)
        print(f"  归位角度: {final_deg:.1f}°  (raw={final_pos})")
        print(f"  [OK] 运动测试完成")

    except Exception as e:
        print(f"  [FAIL] 运动测试异常: {e}")
        # 安全：关闭力矩
        try:
            sdk.write_torque_enable(test_id, False)
        except Exception:
            pass
    finally:
        # 关闭力矩，释放关节
        try:
            sdk.write_torque_enable(test_id, False)
        except Exception:
            pass
        sdk.disconnect()


def main():
    parser = argparse.ArgumentParser(description="机械臂 UART 测试")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"串口设备 (默认 {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"波特率 (默认 {DEFAULT_BAUD})")
    parser.add_argument("--scan-only", action="store_true", help="只扫描，不执行运动")
    parser.add_argument("--move-id", type=int, default=1, help="运动测试的舵机 ID (默认 1)")
    args = parser.parse_args()

    print("=" * 50)
    print("  机械臂 UART 连通性 & 运动测试")
    print("=" * 50)

    # 1. 列出可用串口
    print("\n── 系统可用串口 ──")
    ports = list_serial_ports()
    if ports:
        for p in ports:
            print(f"  {p}")
    else:
        print("  (未发现任何串口设备)")

    # 2. 测试目标串口能否打开
    print(f"\n── 测试串口打开: {args.port} ──")
    if not test_connection(args.port, args.baud):
        print("\n[提示] 请检查:")
        print("  1. 机械臂是否已通过 UART / USB 连接到板子")
        print("  2. 串口设备名是否正确 (--port 参数)")
        print("  3. 当前用户是否有串口权限: sudo usermod -aG dialout $USER")
        sys.exit(1)

    # 3. 扫描舵机
    found_ids = do_scan(args.port, args.baud)

    if not found_ids:
        print("\n[WARN] 未发现任何在线舵机")
        print("[提示] 请检查:")
        print("  1. 舵机是否已上电")
        print("  2. TX/RX 接线是否正确 (需要半双工或交叉)")
        print("  3. 波特率是否匹配 (--baud 参数，默认 1000000)")
        sys.exit(1)

    print(f"\n[OK] 发现 {len(found_ids)} 个舵机: {found_ids}")

    # 4. 读取当前角度
    do_read_positions(args.port, args.baud, found_ids)

    # 5. 运动测试
    if args.scan_only:
        print("\n[--scan-only] 跳过运动测试")
        return

    if args.move_id not in found_ids:
        print(f"\n[WARN] ID={args.move_id} 不在在线列表 {found_ids} 中，跳过运动测试")
        print(f"[提示] 使用 --move-id 指定一个在线舵机 ID")
        return

    print(f"\n[提示] 即将对 ID={args.move_id} 执行 ±10° 来回运动，请确保机械臂有足够运动空间")
    try:
        input("  按 Enter 继续，Ctrl+C 取消 ... ")
    except KeyboardInterrupt:
        print("\n已取消")
        return

    do_move_test(args.port, args.baud, args.move_id)

    print("\n" + "=" * 50)
    print("  测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
