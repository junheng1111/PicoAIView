"""
FeetTech SCS Servo SDK — GPIO UART 版本（树莓派 / 嵌入式 Linux）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
半双工 TTL UART 原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FeetTech 舵机只有一根 DATA 信号线（半双工），
主机发送和接收共用同一根线，有两种接法：

方案 A — 电阻法（推荐，无需额外 GPIO）
    RPi TX ──┬──[1kΩ]── Servo DATA
    RPi RX ──┘
    发送后 echo 字节会出现在 RX，SDK 自动清除。

方案 B — 方向 GPIO 法（干净，长线缆推荐）
    RPi TX ──→ 74HC126/SN74LS241 EN → Servo DATA
    RPi RX ←──────────────────────── Servo DATA
    一个 GPIO 控制使能，发送时 TX 导通 / 接收时高阻。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
树莓派 UART 引脚速查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPIO14 (TXD) — 物理引脚 8
  GPIO15 (RXD) — 物理引脚 10
  设备文件: /dev/serial0 (自动指向可用 UART)
           /dev/ttyAMA0 (Pi 3B/4/5 主 UART)
           /dev/ttyS0   (Pi 3B mini UART，稳定性差)

准备步骤:
  sudo raspi-config → Interface Options → Serial Port
    → "login shell over serial" 选 No
    → "serial port hardware" 选 Yes

依赖:
  pip install pyserial
  pip install RPi.GPIO   # 仅方案 B 需要
"""

import serial
import time
import threading
from typing import Optional, Dict, List, Tuple

# ─── 寄存器地址 ────────────────────────────────────────────────
ADDR_SCS_ID               = 5
ADDR_SCS_BAUD_RATE        = 6
ADDR_MIN_POS_LIMIT        = 9
ADDR_MAX_POS_LIMIT        = 11
ADDR_SCS_MODE             = 33
ADDR_POS_CORRECTION       = 31
ADDR_SCS_TORQUE_ENABLE    = 40
ADDR_SCS_GOAL_ACC         = 41
ADDR_SCS_GOAL_POSITION    = 42
ADDR_SCS_GOAL_SPEED       = 46
ADDR_SCS_PRESENT_POSITION = 56
ADDR_SCS_LOCK             = 55

# ─── 指令集 ────────────────────────────────────────────────────
INST_PING       = 0x01
INST_READ       = 0x02
INST_WRITE      = 0x03
INST_SYNC_WRITE = 0x83

COMM_SUCCESS    = 0
COMM_TX_FAIL    = -2
COMM_RX_TIMEOUT = -6
COMM_RX_CORRUPT = -7


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _lobyte(w: int) -> int:
    return w & 0xFF

def _hibyte(w: int) -> int:
    return (w >> 8) & 0xFF

def _makeword(lo: int, hi: int) -> int:
    return (lo & 0xFF) | ((hi & 0xFF) << 8)

def _checksum(data: List[int]) -> int:
    return (~sum(data)) & 0xFF

def degrees_to_position(degrees: float) -> int:
    return min(int(round(degrees * 4096 / 360)), 4095)

def position_to_degrees(position: int) -> float:
    return (position / 4096) * 360


# ══════════════════════════════════════════════════════════════
# 方案 A：电阻法（无需额外 GPIO，自动清除 echo）
# ══════════════════════════════════════════════════════════════

class ScsServoUART:
    """
    半双工电阻法：TX/RX 通过 1kΩ 电阻并联到舵机 DATA 线。
    发送后自动丢弃 echo，再读取舵机响应。

    接线:
        RPi GPIO14 (TXD, 物理8)  ──┬──[1kΩ]── Servo DATA (黄线)
        RPi GPIO15 (RXD, 物理10) ──┘
        GND ───────────────────────────────── Servo GND  (黑线)
        5V  ───────────────────────────────── Servo VCC  (红线，用独立电源!)
    """

    def __init__(self):
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self.read_timeout = 0.5

    def connect(
        self,
        port: str = "/dev/serial0",
        baud_rate: int = 1_000_000,
    ) -> None:
        """
        连接 GPIO UART

        Args:
            port:      设备文件，树莓派推荐 "/dev/serial0"
            baud_rate: 波特率，默认 1000000
        """
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.read_timeout,
        )
        # 清空缓冲区
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        print(f"[UART] Connected: {port} @ {baud_rate} baud")

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None
        print("[UART] Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def _check(self):
        if not self.is_connected:
            raise ConnectionError("Not connected. Call connect() first.")

    # ── 底层发送 / 接收 ────────────────────────────────────────

    def _send(self, servo_id: int, instruction: int, params: List[int]) -> bytes:
        """
        构建指令包并发送，返回发出的原始字节（用于 echo 计算）
        """
        length = len(params) + 2
        core = [servo_id, length, instruction] + params
        cs = _checksum(core)
        packet = bytes([0xFF, 0xFF] + core + [cs])
        self._serial.reset_input_buffer()
        self._serial.write(packet)
        self._serial.flush()
        return packet

    def _recv(self, expected_params: int) -> Tuple[int, List[int]]:
        """
        先清除 echo（TX 发出的字节会回到 RX），再读取舵机响应。
        Returns:
            (COMM_result, params_list)
        """
        # --- 清除 echo：等 1 个字节传输时间让 echo 到达 ---
        # 已在 _send 前 reset_input_buffer，echo 不会积累
        # 但电阻法下发送的字节会立刻回显，需要读掉
        # 注意：flush() 确保 TX 发完，echo 已在缓冲区
        # 最小响应包: FF FF ID LEN ERR [PARAMS] CS = 6 + expected_params
        expected_len = 6 + expected_params

        raw = self._serial.read(expected_len)
        if len(raw) < 6:
            return COMM_RX_TIMEOUT, []

        data = list(raw)

        # 找包头 0xFF 0xFF
        start = -1
        for i in range(len(data) - 1):
            if data[i] == 0xFF and data[i + 1] == 0xFF:
                start = i
                break
        if start < 0:
            return COMM_RX_CORRUPT, []

        data = data[start:]
        if len(data) < 6:
            return COMM_RX_CORRUPT, []

        pkt_len    = data[3]
        error_byte = data[4]
        params     = data[5 : 4 + pkt_len - 1]
        cs_recv    = data[4 + pkt_len - 1]
        cs_calc    = _checksum(data[2 : 4 + pkt_len - 1])

        if cs_recv != cs_calc:
            return COMM_RX_CORRUPT, []

        return COMM_SUCCESS, params

    def _send_recv(
        self,
        servo_id: int,
        instruction: int,
        params: List[int],
        expected_params: int = 0,
    ) -> Tuple[int, List[int]]:
        """线程安全的发送+接收"""
        self._check()
        with self._lock:
            self._send(servo_id, instruction, params)
            if instruction == INST_SYNC_WRITE:
                return COMM_SUCCESS, []   # SYNC_WRITE 无响应包
            return self._recv(expected_params)

    # ── 基础读写 ──────────────────────────────────────────────

    def _write_1byte(self, servo_id: int, addr: int, val: int):
        self._send_recv(servo_id, INST_WRITE, [addr, val & 0xFF])

    def _write_2byte(self, servo_id: int, addr: int, val: int):
        self._send_recv(servo_id, INST_WRITE, [addr, _lobyte(val), _hibyte(val)])

    def _read_1byte(self, servo_id: int, addr: int) -> int:
        result, params = self._send_recv(servo_id, INST_READ, [addr, 1], 1)
        if result != COMM_SUCCESS or len(params) < 1:
            raise IOError(f"Read 1byte failed: servo={servo_id} addr={addr}")
        return params[0]

    def _read_2byte(self, servo_id: int, addr: int) -> int:
        result, params = self._send_recv(servo_id, INST_READ, [addr, 2], 2)
        if result != COMM_SUCCESS or len(params) < 2:
            raise IOError(f"Read 2byte failed: servo={servo_id} addr={addr}")
        return _makeword(params[0], params[1])

    # ── EEPROM 锁 ─────────────────────────────────────────────

    def unlock_servo(self, servo_id: int):
        self._write_1byte(servo_id, ADDR_SCS_LOCK, 0)

    def lock_servo(self, servo_id: int):
        self._write_1byte(servo_id, ADDR_SCS_LOCK, 1)

    # ── 模式切换 ──────────────────────────────────────────────

    def set_position_mode(self, servo_id: int):
        """切换为位置模式（旋转关节）"""
        self.unlock_servo(servo_id)
        try:
            self._write_1byte(servo_id, ADDR_SCS_MODE, 0)
        finally:
            self.lock_servo(servo_id)

    def set_wheel_mode(self, servo_id: int):
        """切换为轮式模式（连续旋转）"""
        self.unlock_servo(servo_id)
        try:
            self._write_1byte(servo_id, ADDR_SCS_MODE, 1)
        finally:
            self.lock_servo(servo_id)

    # ── 力矩 / 加速度 ─────────────────────────────────────────

    def write_torque_enable(self, servo_id: int, enable: bool):
        self._write_1byte(servo_id, ADDR_SCS_TORQUE_ENABLE, 1 if enable else 0)

    def write_acceleration(self, servo_id: int, acc: int):
        self._write_1byte(servo_id, ADDR_SCS_GOAL_ACC, max(0, min(254, acc)))

    # ── 位置控制 ──────────────────────────────────────────────

    def read_position(self, servo_id: int) -> int:
        """读取原始位置 0~4095"""
        return self._read_2byte(servo_id, ADDR_SCS_PRESENT_POSITION) & 0xFFFF

    def read_position_degrees(self, servo_id: int) -> float:
        return position_to_degrees(self.read_position(servo_id))

    def write_position(self, servo_id: int, position: int):
        """写入目标位置 0~4095"""
        if not (0 <= position <= 4095):
            raise ValueError(f"Position {position} out of range [0,4095]")
        self._write_2byte(servo_id, ADDR_SCS_GOAL_POSITION, round(position))

    def write_position_degrees(self, servo_id: int, degrees: float):
        """写入目标角度 0~360°"""
        if not (0 <= degrees <= 360):
            raise ValueError(f"Degrees {degrees} out of range [0,360]")
        self.write_position(servo_id, degrees_to_position(degrees))

    def sync_read_positions(self, servo_ids: List[int]) -> Dict[int, int]:
        """批量读取位置，返回 {servo_id: raw_position}"""
        result = {}
        for sid in servo_ids:
            try:
                result[sid] = self.read_position(sid)
            except IOError as e:
                print(f"[WARN] {e}")
        return result

    def sync_write_positions(self, servo_positions: Dict[int, int]):
        """批量同步写入位置（一次 SYNC_WRITE 指令）"""
        self._check()
        if not servo_positions:
            return
        params = [ADDR_SCS_GOAL_POSITION, 2]
        for sid, pos in servo_positions.items():
            if not (0 <= pos <= 4095):
                raise ValueError(f"Position {pos} for servo {sid} out of range")
            params += [sid, _lobyte(pos), _hibyte(pos)]
        length = len(params) + 2
        core = [0xFE, length, INST_SYNC_WRITE] + params
        packet = bytes([0xFF, 0xFF] + core + [_checksum(core)])
        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()

    def sync_write_positions_degrees(self, servo_degrees: Dict[int, float]):
        """批量写入角度（度）"""
        self.sync_write_positions({
            sid: degrees_to_position(deg)
            for sid, deg in servo_degrees.items()
        })

    # ── 速度控制（车轮模式）──────────────────────────────────

    def write_wheel_speed(self, servo_id: int, speed: int):
        """写入转速 -10000~10000，负数=反转"""
        speed = max(-10000, min(10000, round(speed)))
        val = abs(speed) & 0x7FFF
        if speed < 0:
            val |= 0x8000
        self._write_2byte(servo_id, ADDR_SCS_GOAL_SPEED, val)

    def sync_write_wheel_speed(self, servo_speeds: Dict[int, int]):
        """批量同步写入转速"""
        self._check()
        if not servo_speeds:
            return
        params = [ADDR_SCS_GOAL_SPEED, 2]
        for sid, speed in servo_speeds.items():
            speed = max(-10000, min(10000, round(speed)))
            val = abs(speed) & 0x7FFF
            if speed < 0:
                val |= 0x8000
            params += [sid, _lobyte(val), _hibyte(val)]
        length = len(params) + 2
        core = [0xFE, length, INST_SYNC_WRITE] + params
        packet = bytes([0xFF, 0xFF] + core + [_checksum(core)])
        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()

    # ── 其他 ──────────────────────────────────────────────────

    def ping(self, servo_id: int) -> bool:
        result, _ = self._send_recv(servo_id, INST_PING, [], 0)
        return result == COMM_SUCCESS


# ══════════════════════════════════════════════════════════════
# 方案 B：GPIO 方向控制法（使用 RPi.GPIO 控制方向引脚）
# ══════════════════════════════════════════════════════════════

class ScsServoUART_DirectionGPIO(ScsServoUART):
    """
    使用一个 GPIO 引脚控制收发方向的半双工 UART。

    接线（以 74HC126 三态缓冲为例）:
        RPi TX  ──→ 74HC126 A  ──→ Servo DATA
        Servo DATA ──→ RPi RX
        RPi DIR_GPIO ──→ 74HC126 OE (高电平=TX 导通，低电平=高阻)

    也可以用简单的 N-MOSFET 电路实现方向控制。
    """

    def __init__(self, dir_pin: int = 18):
        """
        Args:
            dir_pin: BCM 编号的方向控制 GPIO 引脚，默认 GPIO18 (物理引脚12)
        """
        super().__init__()
        self._dir_pin = dir_pin
        self._gpio_available = False

    def connect(
        self,
        port: str = "/dev/serial0",
        baud_rate: int = 1_000_000,
        dir_pin: Optional[int] = None,
    ) -> None:
        if dir_pin is not None:
            self._dir_pin = dir_pin
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._dir_pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio_available = True
            print(f"[UART] Direction GPIO{self._dir_pin} initialized")
        except ImportError:
            print("[WARN] RPi.GPIO not found, falling back to echo method")
            self._gpio_available = False

        super().connect(port, baud_rate)

    def disconnect(self) -> None:
        super().disconnect()
        if self._gpio_available:
            self._GPIO.cleanup(self._dir_pin)

    def _set_tx_mode(self):
        """切换到发送模式（DIR 引脚高电平）"""
        if self._gpio_available:
            self._GPIO.output(self._dir_pin, self._GPIO.HIGH)
            time.sleep(0.000050)  # 50µs 建立时间

    def _set_rx_mode(self):
        """切换到接收模式（DIR 引脚低电平）"""
        if self._gpio_available:
            time.sleep(0.000050)  # 等待最后一个字节发完
            self._GPIO.output(self._dir_pin, self._GPIO.LOW)

    def _send_recv(
        self,
        servo_id: int,
        instruction: int,
        params: List[int],
        expected_params: int = 0,
    ) -> Tuple[int, List[int]]:
        self._check()
        with self._lock:
            self._set_tx_mode()
            self._send(servo_id, instruction, params)
            self._set_rx_mode()

            if instruction == INST_SYNC_WRITE:
                return COMM_SUCCESS, []
            return self._recv(expected_params)

    def sync_write_positions(self, servo_positions: Dict[int, int]):
        self._check()
        if not servo_positions:
            return
        params = [ADDR_SCS_GOAL_POSITION, 2]
        for sid, pos in servo_positions.items():
            params += [sid, _lobyte(pos), _hibyte(pos)]
        length = len(params) + 2
        core = [0xFE, length, INST_SYNC_WRITE] + params
        packet = bytes([0xFF, 0xFF] + core + [_checksum(core)])
        with self._lock:
            self._set_tx_mode()
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()
            self._set_rx_mode()

    def sync_write_wheel_speed(self, servo_speeds: Dict[int, int]):
        self._check()
        if not servo_speeds:
            return
        params = [ADDR_SCS_GOAL_SPEED, 2]
        for sid, speed in servo_speeds.items():
            speed = max(-10000, min(10000, round(speed)))
            val = abs(speed) & 0x7FFF
            if speed < 0:
                val |= 0x8000
            params += [sid, _lobyte(val), _hibyte(val)]
        length = len(params) + 2
        core = [0xFE, length, INST_SYNC_WRITE] + params
        packet = bytes([0xFF, 0xFF] + core + [_checksum(core)])
        with self._lock:
            self._set_tx_mode()
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()
            self._set_rx_mode()
