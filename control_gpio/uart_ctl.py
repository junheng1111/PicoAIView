#!/usr/bin/env python3
"""
GPIO UART (40PIN Pin8 TX / Pin10 RX) 工具类
用于通过 UART 与机械臂 / 舵机 / 外部 MCU 通信
"""
import serial
import time
import threading
from typing import Optional, Callable


class UartControl:
    """GPIO UART 通信控制类"""

    def __init__(
        self,
        port: str = "/dev/ttyS1",  # RDK X5 上 40PIN UART 对应 ttyS1
        baud: int = 1000000,
        timeout: float = 0.5,
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None

    # ── 基础操作 ──────────────────────────────────

    def open(self) -> bool:
        """打开串口"""
        try:
            self.ser = serial.Serial(
                self.port, self.baud,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"[OK] 打开 {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"[FAIL] 打开 {self.port} 失败: {e}")
            return False

    def close(self):
        """关闭串口"""
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[OK] 关闭 {self.port}")

    @property
    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ── 发送 / 接收 ──────────────────────────────────

    def send(self, data: bytes) -> int:
        """发送字节数据，返回发送字节数"""
        if not self.is_open:
            print("[FAIL] 串口未打开")
            return 0
        n = self.ser.write(data)
        self.ser.flush()
        return n

    def send_hex(self, hex_str: str):
        """发送十六进制字符串，如 'FF FF FE 04 02 01 00 FC'"""
        data = bytes.fromhex(hex_str.replace(" ", ""))
        self.send(data)

    def send_text(self, text: str):
        """发送文本（自动加换行）"""
        self.send((text + "\n").encode("utf-8"))

    def recv(self, size: int = 256) -> bytes:
        """读取数据"""
        if not self.is_open:
            return b""
        return self.ser.read(size)

    def recv_all(self, wait: float = 0.3) -> bytes:
        """等待一小段时间后读取所有数据"""
        time.sleep(wait)
        data = b""
        while self.ser.in_waiting:
            data += self.ser.read(self.ser.in_waiting)
        return data

    # ── 持续监听（后台线程） ──────────────────────────

    def start_monitor(self, callback: Optional[Callable[[bytes], None]] = None):
        """后台持续监听串口数据"""
        if self._running:
            print("[WARN] 已经在监听中")
            return
        self._running = True

        def _loop():
            while self._running and self.is_open:
                try:
                    data = self.ser.read(256)
                    if data:
                        if callback:
                            callback(data)
                        else:
                            # 默认输出：尝试文本，否则显示 hex
                            try:
                                print(data.decode("utf-8"), end="", flush=True)
                            except Exception:
                                print(data.hex(), flush=True)
                except Exception:
                    break

        self._recv_thread = threading.Thread(target=_loop, daemon=True)
        self._recv_thread.start()
        print(f"[OK] 开始监听 {self.port}")

    def stop_monitor(self):
        """停止后台监听"""
        self._running = False
        print("[OK] 停止监听")

    # ── 机械臂常用协议 ──────────────────────────────

    def dynamixel_ping(self, servo_id: int = 1) -> Optional[bytes]:
        """
        Dynamixel 协议 ping 舵机
        指令: FF FF ID 04 02 01 CHK
        """
        if not self.is_open:
            return None
        cmd = bytes([0xFF, 0xFF, servo_id, 0x04, 0x02, 0x01])
        chk = 0
        for b in cmd[2:]:
            chk ^= b
        cmd += bytes([chk ^ 0xFF & 0xFF])
        self.ser.write(cmd)
        time.sleep(0.1)
        resp = self.ser.read(256)
        return resp if resp else None

    def dynamixel_write(self, servo_id: int, addr: int, data: bytes):
        """Dynamixel 协议写寄存器"""
        length = len(data) + 3
        cmd = bytes([0xFF, 0xFF, servo_id, length, 0x03, addr & 0xFF])
        cmd += data
        chk = 0
        for b in cmd[2:]:
            chk ^= b
        cmd += bytes([chk ^ 0xFF & 0xFF])
        self.send(cmd)

    def dynamixel_read(self, servo_id: int, addr: int, length: int) -> Optional[bytes]:
        """Dynamixel 协议读寄存器"""
        cmd = bytes([0xFF, 0xFF, servo_id, 0x04, 0x02, addr & 0xFF, length & 0xFF])
        chk = 0
        for b in cmd[2:]:
            chk ^= b
        cmd += bytes([chk ^ 0xFF & 0xFF])
        self.send(cmd)
        time.sleep(0.1)
        return self.ser.read(256) if self.is_open else None

    # ── 工具方法 ──────────────────────────────────

    def probe(self, baud_list=None):
        """多波特率探测，看看哪个波特率有响应"""
        if baud_list is None:
            baud_list = [9600, 57600, 115200, 460800, 1000000, 1500000]

        print(f"\n=== 多波特率探测 {self.port} ===")
        original_baud = self.baud
        for baud in baud_list:
            try:
                self.close()
                self.baud = baud
                if not self.open():
                    continue
                # 发一个 Dynamixel ping (ID=0xFE 广播)
                self.send(bytes([0xFF, 0xFF, 0xFE, 0x04, 0x02, 0x01, 0xFC]))
                time.sleep(0.15)
                resp = self.recv()
                if resp:
                    print(f"  [{baud:>7}] ✅ 收到响应! ({len(resp)} bytes) hex={resp.hex()}")
                else:
                    print(f"  [{baud:>7}] ❌ 无响应")
            except Exception as e:
                print(f"  [{baud:>7}] ❌ 错误: {e}")

        self.close()
        self.baud = original_baud
        self.open()
        print("=== 探测结束 ===\n")


# ── 交互式测试 ─────────────────────────────────

def interactive_test():
    """交互式测试 UART 通信"""
    uart = UartControl()

    if not uart.open():
        return

    print("\n=== GPIO UART 交互测试 ===")
    print("可用命令:")
    print("  send <hex>       - 发送十六进制，如: send FF FF FE 04 02 01 FC")
    print("  text <msg>       - 发送文本")
    print("  recv             - 接收一次")
    print("  monitor          - 启动后台监听")
    print("  stop             - 停止监听")
    print("  probe            - 多波特率探测")
    print("  ping [id]        - Dynamixel ping (默认 ID=1)")
    print("  baud <num>       - 切换波特率")
    print("  hex              - 切换 Hex 显示模式")
    print("  status           - 查看串口状态")
    print("  q                - 退出")
    print("=" * 40)

    try:
        while True:
            cmd = input("uart> ").strip()
            if not cmd:
                continue
            if cmd == "q":
                break
            elif cmd == "recv":
                data = uart.recv_all()
                if data:
                    print(f"收到 ({len(data)}): {data.hex()}")
                else:
                    print("无数据")
            elif cmd == "monitor":
                uart.start_monitor()
            elif cmd == "stop":
                uart.stop_monitor()
            elif cmd == "probe":
                uart.probe()
            elif cmd == "status":
                print(f"串口: {uart.port}, 波特率: {uart.baud}, 打开: {uart.is_open}")
            elif cmd.startswith("send "):
                hex_str = cmd[5:].strip()
                uart.send_hex(hex_str)
                print(f"已发送: {hex_str}")
            elif cmd.startswith("text "):
                text = cmd[5:].strip()
                uart.send_text(text)
                print(f"已发送文本: {text}")
            elif cmd.startswith("ping"):
                parts = cmd.split()
                sid = int(parts[1]) if len(parts) > 1 else 1
                resp = uart.dynamixel_ping(sid)
                if resp:
                    print(f"Ping ID={sid} 成功: {resp.hex()}")
                else:
                    print(f"Ping ID={sid} 无响应")
            elif cmd.startswith("baud "):
                b = int(cmd.split()[1])
                uart.close()
                uart.baud = b
                uart.open()
            else:
                print("未知命令")
    except KeyboardInterrupt:
        pass
    finally:
        uart.stop_monitor()
        uart.close()
    print("已退出")


if __name__ == "__main__":
    # 默认运行交互测试
    interactive_test()
