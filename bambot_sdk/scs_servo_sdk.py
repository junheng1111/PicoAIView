"""
FeetTech SCS Servo SDK - Python 版本
对应 feetech.js/scsServoSDK.mjs 的完整 Python 实现

依赖: pip install pyserial
协议: FeetTech STS/SMS (protocolEnd=0, 小端字节序)

数据包格式:
  [0xFF] [0xFF] [ID] [LENGTH] [INSTRUCTION] [PARAM0...] [CHECKSUM]
  CHECKSUM = ~(ID + LENGTH + INSTRUCTION + PARAM...) & 0xFF
"""

import serial
import time
from typing import Optional, Dict, List, Tuple

# ─── 寄存器地址 ────────────────────────────────────────────────
ADDR_SCS_ID              = 5
ADDR_SCS_BAUD_RATE       = 6
ADDR_MIN_POS_LIMIT       = 9
ADDR_MAX_POS_LIMIT       = 11
ADDR_SCS_MODE            = 33
ADDR_POS_CORRECTION      = 31
ADDR_SCS_TORQUE_ENABLE   = 40
ADDR_SCS_GOAL_ACC        = 41
ADDR_SCS_GOAL_POSITION   = 42
ADDR_SCS_GOAL_SPEED      = 46
ADDR_SCS_PRESENT_POSITION = 56
ADDR_SCS_LOCK            = 55

# ─── 指令集 ────────────────────────────────────────────────────
INST_PING       = 0x01
INST_READ       = 0x02
INST_WRITE      = 0x03
INST_SYNC_WRITE = 0x83

# ─── 通信结果 ──────────────────────────────────────────────────
COMM_SUCCESS    = 0
COMM_TX_FAIL    = -2
COMM_RX_FAIL    = -3
COMM_RX_TIMEOUT = -6
COMM_RX_CORRUPT = -7

# ─── 协议字节序工具（protocolEnd=0，STS/SMS 小端序） ──────────
def _lobyte(w: int) -> int:
    return w & 0xFF

def _hibyte(w: int) -> int:
    return (w >> 8) & 0xFF

def _makeword(lo: int, hi: int) -> int:
    return (lo & 0xFF) | ((hi & 0xFF) << 8)

def _checksum(packet_bytes: List[int]) -> int:
    """计算校验和: ~(ID + LENGTH + INST + PARAMS...) & 0xFF"""
    return (~sum(packet_bytes)) & 0xFF

# ─── 单位换算 ──────────────────────────────────────────────────
def degrees_to_position(degrees: float) -> int:
    """角度 (0~360°) → 舵机位置值 (0~4095)"""
    return min(int(round(degrees * 4096 / 360)), 4095)

def position_to_degrees(position: int) -> float:
    """舵机位置值 (0~4095) → 角度 (0~360°)"""
    return (position / 4096) * 360


class ScsServoSDK:
    """
    FeetTech SCS 舵机控制 SDK (Python)

    用法示例:
        sdk = ScsServoSDK()
        sdk.connect("COM3")               # Windows
        sdk.connect("/dev/ttyUSB0")       # Linux

        sdk.set_position_mode(1)
        sdk.write_torque_enable(1, True)
        sdk.write_position(1, 2048)       # 原始位置值
        sdk.write_position_degrees(1, 180.0)  # 角度接口

        sdk.set_wheel_mode(2)
        sdk.write_wheel_speed(2, 300)     # 正转
        sdk.write_wheel_speed(2, -300)    # 反转

        sdk.disconnect()
    """

    def __init__(self, half_duplex: bool = False):
        self._serial: Optional[serial.Serial] = None
        self.timeout = 0.5  # 读取超时秒数
        self.half_duplex = half_duplex  # TX/RX 短接时需要丢弃回声

    # ══════════════════════════════════════════════════════════
    # 连接 / 断开
    # ══════════════════════════════════════════════════════════

    def connect(self, port: str, baud_rate: int = 1_000_000) -> None:
        """
        打开串口连接

        Args:
            port:      串口名, Windows 如 "COM3", Linux 如 "/dev/ttyUSB0"
            baud_rate: 波特率, 默认 1000000
        """
        if self._serial and self._serial.is_open:
            print("Already connected.")
            return
        self._serial = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
        )
        print(f"Connected to {port} at {baud_rate} baud.")

    def disconnect(self) -> None:
        """关闭串口连接"""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("Disconnected.")
        self._serial = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def _check_connection(self):
        if not self.is_connected:
            raise ConnectionError("Not connected. Call connect() first.")

    # ══════════════════════════════════════════════════════════
    # 底层读写（私有）
    # ══════════════════════════════════════════════════════════

    def _send_packet(self, servo_id: int, instruction: int, params: List[int]) -> None:
        """构建并发送指令包（匹配 JS txPacket + txRxPacket 逻辑）"""
        length = len(params) + 2  # INSTRUCTION + PARAMS + CHECKSUM
        packet_core = [servo_id, length, instruction] + params
        checksum = _checksum(packet_core)
        packet = bytes([0xFF, 0xFF] + packet_core + [checksum])
        self._serial.reset_input_buffer()   # 发送前清空（JS: clearPort before write）
        self._serial.write(packet)
        self._serial.flush()                # 等待 TX 完成
        # 发送后再清空：丢弃半双工回声（JS: clearPort after write, before rxPacket）
        # 用 reset_input_buffer 替代 read(n)，避免因无回声而卡 timeout
        if self.half_duplex:
            time.sleep(len(packet) * 10 / self._serial.baudrate + 0.001)
            self._serial.reset_input_buffer()

    def _recv_packet(self, expected_params: int = 0) -> Tuple[int, List[int], int]:
        """
        接收状态包
        Returns:
            (comm_result, params, error_byte)
        """
        # 最小包长: FF FF ID LEN ERR ... CHECKSUM = 6 bytes
        expected_len = 6 + expected_params
        raw = self._serial.read(expected_len)
        if len(raw) < 6:
            return COMM_RX_TIMEOUT, [], 0

        data = list(raw)

        # 找包头
        header = -1
        for i in range(len(data) - 1):
            if data[i] == 0xFF and data[i + 1] == 0xFF:
                header = i
                break
        if header < 0:
            return COMM_RX_CORRUPT, [], 0

        data = data[header:]
        if len(data) < 6:
            return COMM_RX_CORRUPT, [], 0

        pkt_id     = data[2]
        pkt_len    = data[3]
        error_byte = data[4]
        params     = data[5 : 4 + pkt_len - 1]
        checksum   = data[4 + pkt_len - 1]

        # 校验
        calc_cs = _checksum(data[2 : 4 + pkt_len - 1])
        if checksum != calc_cs:
            return COMM_RX_CORRUPT, [], error_byte

        return COMM_SUCCESS, params, error_byte

    def _write_1byte(self, servo_id: int, address: int, value: int) -> None:
        self._check_connection()
        self._send_packet(servo_id, INST_WRITE, [address, value & 0xFF])
        self._recv_packet(0)  # 消耗响应包

    def _write_2byte(self, servo_id: int, address: int, value: int) -> None:
        self._check_connection()
        params = [address, _lobyte(value), _hibyte(value)]
        self._send_packet(servo_id, INST_WRITE, params)
        self._recv_packet(0)

    def _read_1byte(self, servo_id: int, address: int) -> int:
        self._check_connection()
        self._send_packet(servo_id, INST_READ, [address, 1])
        result, params, _ = self._recv_packet(1)
        if result != COMM_SUCCESS or len(params) < 1:
            raise IOError(f"Failed to read 1 byte from servo {servo_id} addr {address}")
        return params[0]

    def _read_2byte(self, servo_id: int, address: int) -> int:
        self._check_connection()
        self._send_packet(servo_id, INST_READ, [address, 2])
        result, params, _ = self._recv_packet(2)
        if result != COMM_SUCCESS or len(params) < 2:
            raise IOError(f"Failed to read 2 bytes from servo {servo_id} addr {address}")
        return _makeword(params[0], params[1])

    # ══════════════════════════════════════════════════════════
    # EEPROM 锁（修改配置前需解锁）
    # ══════════════════════════════════════════════════════════

    def unlock_servo(self, servo_id: int) -> None:
        """解锁 EEPROM（写配置寄存器前必须调用）"""
        self._write_1byte(servo_id, ADDR_SCS_LOCK, 0)

    def lock_servo(self, servo_id: int) -> None:
        """锁定 EEPROM"""
        self._write_1byte(servo_id, ADDR_SCS_LOCK, 1)

    # ══════════════════════════════════════════════════════════
    # 模式切换
    # ══════════════════════════════════════════════════════════

    def set_position_mode(self, servo_id: int) -> None:
        """切换为位置模式（旋转关节，mode=0）"""
        self.unlock_servo(servo_id)
        try:
            self._write_1byte(servo_id, ADDR_SCS_MODE, 0)
        finally:
            self.lock_servo(servo_id)
        print(f"Servo {servo_id} → position mode")

    def set_wheel_mode(self, servo_id: int) -> None:
        """切换为轮式模式（连续旋转，mode=1）"""
        self.unlock_servo(servo_id)
        try:
            self._write_1byte(servo_id, ADDR_SCS_MODE, 1)
        finally:
            self.lock_servo(servo_id)
        print(f"Servo {servo_id} → wheel mode")

    def read_mode(self, servo_id: int) -> int:
        """读取当前模式 (0=位置, 1=轮式)"""
        return self._read_1byte(servo_id, ADDR_SCS_MODE)

    # ══════════════════════════════════════════════════════════
    # 力矩 / 加速度
    # ══════════════════════════════════════════════════════════

    def write_torque_enable(self, servo_id: int, enable: bool) -> None:
        """启用或关闭舵机力矩"""
        self._write_1byte(servo_id, ADDR_SCS_TORQUE_ENABLE, 1 if enable else 0)

    def write_acceleration(self, servo_id: int, acceleration: int) -> None:
        """设置加速度 (0~254)"""
        acc = max(0, min(254, acceleration))
        self._write_1byte(servo_id, ADDR_SCS_GOAL_ACC, acc)

    # ══════════════════════════════════════════════════════════
    # 位置控制（revolute 关节）
    # ══════════════════════════════════════════════════════════

    def read_position(self, servo_id: int) -> int:
        """读取当前位置 (0~4095)"""
        return self._read_2byte(servo_id, ADDR_SCS_PRESENT_POSITION) & 0xFFFF

    def read_position_degrees(self, servo_id: int) -> float:
        """读取当前角度 (0~360°)"""
        return position_to_degrees(self.read_position(servo_id))

    def write_position(self, servo_id: int, position: int) -> None:
        """
        写入目标位置 (0~4095)

        Args:
            servo_id: 舵机 ID
            position: 目标位置值 0~4095
        """
        if not (0 <= position <= 4095):
            raise ValueError(f"Position {position} out of range [0, 4095]")
        self._write_2byte(servo_id, ADDR_SCS_GOAL_POSITION, round(position))

    def write_position_degrees(self, servo_id: int, degrees: float) -> None:
        """
        写入目标角度 (0~360°)

        Args:
            servo_id: 舵机 ID
            degrees:  目标角度，0~360
        """
        if not (0 <= degrees <= 360):
            raise ValueError(f"Degrees {degrees} out of range [0, 360]")
        self.write_position(servo_id, degrees_to_position(degrees))

    def sync_read_positions(self, servo_ids: List[int]) -> Dict[int, int]:
        """
        批量读取多个舵机位置

        Returns:
            {servo_id: position(0~4095), ...}
        """
        result = {}
        for sid in servo_ids:
            try:
                result[sid] = self.read_position(sid)
            except IOError as e:
                print(f"[WARN] sync_read_positions: {e}")
        return result

    def sync_write_positions(self, servo_positions: Dict[int, int]) -> None:
        """
        批量写入多个舵机位置（一次 SYNC_WRITE 指令）

        Args:
            servo_positions: {servo_id: position(0~4095), ...}
        """
        self._check_connection()
        if not servo_positions:
            return

        data_len = 2          # 每个舵机 2 字节位置
        params = [ADDR_SCS_GOAL_POSITION, data_len]
        for sid, pos in servo_positions.items():
            if not (0 <= pos <= 4095):
                raise ValueError(f"Position {pos} for servo {sid} out of range")
            params += [sid, _lobyte(pos), _hibyte(pos)]

        length = len(params) + 2
        packet_core = [0xFE, length, INST_SYNC_WRITE] + params
        checksum = _checksum(packet_core)
        packet = bytes([0xFF, 0xFF] + packet_core + [checksum])
        self._serial.reset_input_buffer()
        self._serial.write(packet)
        # SYNC_WRITE 无响应包

    def sync_write_positions_degrees(self, servo_degrees: Dict[int, float]) -> None:
        """
        批量写入多个舵机角度（度）

        Args:
            servo_degrees: {servo_id: degrees(0~360), ...}
        """
        servo_positions = {
            sid: degrees_to_position(deg)
            for sid, deg in servo_degrees.items()
        }
        self.sync_write_positions(servo_positions)

    # ══════════════════════════════════════════════════════════
    # 速度控制（continuous / 轮式）
    # ══════════════════════════════════════════════════════════

    def write_wheel_speed(self, servo_id: int, speed: int) -> None:
        """
        写入轮式转速

        Args:
            servo_id: 舵机 ID
            speed:    转速 -10000~10000，负数=反转
                      编码: bit15=方向位，bits0-14=速度大小
        """
        speed = max(-10000, min(10000, round(speed)))
        speed_val = abs(speed) & 0x7FFF
        if speed < 0:
            speed_val |= 0x8000
        self._write_2byte(servo_id, ADDR_SCS_GOAL_SPEED, speed_val)

    def sync_write_wheel_speed(self, servo_speeds: Dict[int, int]) -> None:
        """
        批量写入多个舵机转速（一次 SYNC_WRITE 指令）

        Args:
            servo_speeds: {servo_id: speed(-10000~10000), ...}
        """
        self._check_connection()
        if not servo_speeds:
            return

        data_len = 2
        params = [ADDR_SCS_GOAL_SPEED, data_len]
        for sid, speed in servo_speeds.items():
            speed = max(-10000, min(10000, round(speed)))
            speed_val = abs(speed) & 0x7FFF
            if speed < 0:
                speed_val |= 0x8000
            params += [sid, _lobyte(speed_val), _hibyte(speed_val)]

        length = len(params) + 2
        packet_core = [0xFE, length, INST_SYNC_WRITE] + params
        checksum = _checksum(packet_core)
        packet = bytes([0xFF, 0xFF] + packet_core + [checksum])
        self._serial.reset_input_buffer()
        self._serial.write(packet)

    # ══════════════════════════════════════════════════════════
    # 位置偏移校正
    # ══════════════════════════════════════════════════════════

    def read_pos_correction(self, servo_id: int) -> int:
        """读取位置偏移 (-2047~2047)"""
        raw = self._read_2byte(servo_id, ADDR_POS_CORRECTION)
        magnitude = raw & 0x7FF
        direction = -1 if (raw & 0x800) else 1
        return direction * magnitude

    def write_pos_correction(self, servo_id: int, correction: int) -> None:
        """
        写入位置偏移 (-2047~2047)
        注意：会自动 unlock/lock EEPROM
        """
        if not (-2047 <= correction <= 2047):
            raise ValueError(f"Correction {correction} out of range [-2047, 2047]")
        value = abs(correction) & 0x7FF
        if correction < 0:
            value |= 0x800
        self.unlock_servo(servo_id)
        try:
            self._write_2byte(servo_id, ADDR_POS_CORRECTION, value)
        finally:
            self.lock_servo(servo_id)

    # ══════════════════════════════════════════════════════════
    # 其他读取
    # ══════════════════════════════════════════════════════════

    def read_baud_rate(self, servo_id: int) -> int:
        """读取波特率索引"""
        return self._read_1byte(servo_id, ADDR_SCS_BAUD_RATE)

    def ping(self, servo_id: int) -> bool:
        """Ping 单个舵机，返回是否在线"""
        self._check_connection()
        self._send_packet(servo_id, INST_PING, [])
        result, _, _ = self._recv_packet(0)
        return result == COMM_SUCCESS


# ─── 快捷工具函数 ──────────────────────────────────────────────

def scan_servos(port: str, baud_rate: int = 1_000_000,
                id_range: range = range(1, 20),
                half_duplex: bool = True) -> List[int]:
    """
    扫描在线舵机 ID

    Args:
        port:        串口
        baud_rate:   波特率
        id_range:    扫描范围，默认 1~19
        half_duplex: TX/RX 短接的半双工模式（默认 True）
    Returns:
        在线舵机 ID 列表
    """
    sdk = ScsServoSDK(half_duplex=half_duplex)
    sdk.connect(port, baud_rate)
    online = []
    for sid in id_range:
        if sdk.ping(sid):
            online.append(sid)
            print(f"  Found servo ID: {sid}")
    sdk.disconnect()
    return online
