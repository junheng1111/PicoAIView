# BamBot 控制架构总结

## 一、整体调用链

```
用户输入 (键盘 / AI 聊天)
    │
    ▼
KeyboardControl / ChatControl  (React 组件)
    │
    ▼
useRobotControl Hook           (状态管理 + 业务逻辑)
    │
    ▼
ScsServoSDK                    (feetech.js — 串口协议层)
    │
    ▼
Web Serial API                 (浏览器原生串口)
    │
    ▼
FeetTech SCS 伺服电机           (物理设备, ID 1~252)
```

---

## 二、底层电机控制 SDK（`feetech.js/scsServoSDK.mjs`）

### 2.1 寄存器地址表

| 名称 | 地址 | 说明 |
|------|------|------|
| `ADDR_SCS_TORQUE_ENABLE` | 40 | 力矩使能 (0=关, 1=开) |
| `ADDR_SCS_GOAL_ACC` | 41 | 目标加速度 (0~254) |
| `ADDR_SCS_GOAL_POSITION` | 42 | 目标位置 (0~4095) |
| `ADDR_SCS_GOAL_SPEED` | 46 | 目标速度 (bit15=方向) |
| `ADDR_SCS_PRESENT_POSITION` | 56 | 当前位置 (只读) |
| `ADDR_SCS_MODE` | 33 | 模式 (0=位置, 1=轮式) |
| `ADDR_SCS_LOCK` | 55 | EEPROM 锁 (0=解锁, 1=锁定) |
| `ADDR_POS_CORRECTION` | 31 | 位置偏移校正 |
| `ADDR_MIN_POS_LIMIT` | 9 | 最小位置限制 |
| `ADDR_MAX_POS_LIMIT` | 11 | 最大位置限制 |

### 2.2 连接接口

```js
// 连接串口（默认波特率 1000000）
await sdk.connect({ baudRate: 1000000, protocolEnd: 0 })

// 断开串口
await sdk.disconnect()

// 检查连接状态（未连接则抛出异常）
sdk.checkConnection()
```

### 2.3 模式切换接口

```js
// 切换为轮式模式（连续旋转，用于底盘车轮）
await sdk.setWheelMode(servoId)

// 切换为位置模式（精确角度，用于机械臂关节）
await sdk.setPositionMode(servoId)
```

### 2.4 位置控制接口（关节 / revolute）

```js
// 读取当前位置（返回 0~4095）
const rawPos = await sdk.readPosition(servoId)

// 写入目标位置（0~4095，对应 0°~360°）
await sdk.writePosition(servoId, position)

// 批量读取多个舵机位置（返回 Map<servoId, position>）
const positions = await sdk.syncReadPositions([1, 2, 3])

// 批量写入多个舵机位置（同步写，一次发包）
await sdk.syncWritePositions({ 1: 2048, 2: 1024 })
```

**单位换算：**
$$\text{position} = \left\lfloor \frac{\text{degrees} \times 4096}{360} \right\rfloor$$
$$\text{degrees} = \frac{\text{position}}{4096} \times 360$$

### 2.5 速度控制接口（车轮 / continuous）

```js
// 写入单个舵机转速（-10000~10000，负数=反转）
// 编码规则：bit15=方向位，bits0-14=速度大小
await sdk.writeWheelSpeed(servoId, speed)

// 批量写入多个舵机速度（同步写）
await sdk.syncWriteWheelSpeed({ 13: 300, 14: -300, 15: 300 })
```

### 2.6 力矩 / 加速度接口

```js
// 启用/禁用力矩（true=启用）
await sdk.writeTorqueEnable(servoId, true)

// 设置加速度（0~254）
await sdk.writeAcceleration(servoId, 50)
```

### 2.7 EEPROM 锁 / 配置写入

```js
// 解锁 EEPROM（写配置前必须调用）
await sdk.unlockServo(servoId)

// 锁定 EEPROM
await sdk.lockServo(servoId)

// 读/写位置偏移校正（-2047~2047）
const correction = await sdk.readPosCorrection(servoId)
await sdk.writePosCorrection(servoId, 100)

// 批量读写位置校正
const map = await sdk.syncReadPosCorrection([1, 2, 3])
await sdk.syncWritePosCorrection({ 1: 50, 2: -30 })
```

---

## 三、React 业务层（`hooks/useRobotControl.ts`）

### 3.1 连接管理

```ts
const {
  isConnected,
  connectRobot,      // 连接并初始化所有关节
  disconnectRobot,   // 停止所有电机并断开连接
} = useRobotControl(initialJointDetails, urdfInitJointAngles)
```

**`connectRobot` 初始化流程：**
1. 调用 `sdk.connect()` 打开串口
2. 对每个关节：
   - `revolute`（旋转关节）→ `setPositionMode` → `readPosition` → `writeTorqueEnable(true)`
   - `continuous`（车轮）→ `setWheelMode` → 速度置 0

**`disconnectRobot` 流程：**
1. 车轮速度置 0：`writeWheelSpeed(id, 0)`
2. 所有关节力矩关闭：`writeTorqueEnable(id, false)`
3. `sdk.disconnect()`

### 3.2 关节状态类型

```ts
type JointState = {
  name: string
  servoId?: number
  jointType: "revolute" | "continuous"
  limit?: { lower?: number; upper?: number }
  degrees?: number | "N/A" | "error"   // revolute 关节用
  speed?: number | "N/A" | "error"     // continuous 关节用
}
```

### 3.3 运动控制接口

```ts
// 单关节角度控制（revolute，0~360°）
updateJointDegrees(servoId: number, degrees: number): Promise<void>

// 单关节速度控制（continuous，-10000~10000）
updateJointSpeed(servoId: number, speed: number): Promise<void>

// 批量角度控制（内部使用 syncWritePositions，单次串口包）
updateJointsDegrees(updates: { servoId: number; value: number }[]): Promise<void>

// 批量速度控制（内部使用 syncWriteWheelSpeed）
updateJointsSpeed(updates: { servoId: number; speed: number }[]): Promise<void>
```

### 3.4 录制回放接口

```ts
startRecording()   // 开始录制（每 20ms 记录一帧所有关节状态）
stopRecording()    // 停止录制
clearRecordData()  // 清空录制数据
recordData: RecordData  // number[][] 每帧为所有关节的 degrees/speed 数组
```

---

## 四、机器人配置（`config/robotConfig.ts`）

每个机器人型号的配置包含：

| 字段 | 说明 |
|------|------|
| `urdfUrl` | 3D 模型文件路径 |
| `jointNameIdMap` | URDF 关节名 → 舵机 ID 映射 |
| `keyboardControlMap` | 舵机 ID → 键位数组映射 |
| `compoundMovements` | 联动关节配置（一个按键同时驱动多个关节） |
| `systemPrompt` | 给 AI 的系统提示词 |

**典型配置示例（so-arm100 机械臂）：**

```ts
keyboardControlMap: {
  1: ["1", "q"],   // Rotation 关节
  2: ["2", "w"],   // Pitch 关节
  3: ["3", "e"],   // Elbow 关节
  4: ["4", "r"],   // Wrist_Pitch 关节
  5: ["5", "t"],   // Wrist_Roll 关节
  6: ["6", "y"],   // Jaw（夹爪）
}
```

**底盘配置（bambot-b0-base）：**

```ts
// 使用方向键控制三轮底盘
// ArrowUp=前进, ArrowDown=后退, ArrowLeft=左转, ArrowRight=右转
// 车轮 servoId: 13(左), 14(后), 15(右)
```

---

## 五、AI 控制机制（`components/playground/chatControl/ChatControl.tsx`）

### 5.1 整体流程

```
用户自然语言输入
    │
    ▼
generateText(OpenAI API)
    │  system prompt: 告知 AI 可用按键及含义
    │  tools: { keyPress }
    │
    ▼
AI 决策 → 调用 keyPress(key, duration)
    │
    ▼
模拟 KeyboardEvent (keydown → wait → keyup)
    │
    ▼
KeyboardControl 监听事件 → updateJointDegrees / updateJointSpeed
    │
    ▼
ScsServoSDK → 串口 → 电机运动
```

### 5.2 `keyPress` 工具定义

```ts
keyPress: tool({
  description: "Press and hold a keyboard key for a specified duration (ms) to control the robot",
  parameters: z.object({
    key: z.string().describe(
      "The key to press (e.g., 'w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight')"
    ),
    duration: z.number().int().min(100).max(5000).default(1000)
      .describe("How long to hold the key in milliseconds (default: 1000)"),
  }),
  execute: async ({ key, duration }) => {
    window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }))
    await sleep(duration)
    window.dispatchEvent(new KeyboardEvent("keyup", { key, bubbles: true }))
    return `Held key "${key}" for ${duration} ms`
  }
})
```

### 5.3 各机器人的 System Prompt

**so-arm100 机械臂：**
```
You can help control the so-arm100 robot by pressing keyboard keys.
Use the keyPress tool to simulate key presses.
Each key will be held down for 1 second by default.
The robot can be controlled with the following keys:
- "q" and "1" for rotate the bot to left and right
- "i" and "8" for moving the bot/jaw down("i") and up("8")
- "u" and "o" for moving the bot/jaw backward("u") and forward("o")
- "6" to open the jaw and "y" to close the jaw
- "t" and "5" for rotating jaw
```

**bambot-b0-base 底盘：**
```
You can help control the bambot-b0-base robot by pressing keyboard keys.
- "ArrowUp" to move forward
- "ArrowDown" to move backward
- "ArrowLeft" to turn left
- "ArrowRight" to turn right
```

**默认 prompt（无配置时）：**
```
You can help control a robot by pressing keyboard keys.
Use the keyPress tool to simulate key presses.
Each key will be held down for 1 second by default.
```

### 5.4 AI 设置项（存储于 localStorage）

| 配置项 | 键名 | 说明 |
|--------|------|------|
| API Key | `api_key` | OpenAI 兼容 API 密钥 |
| Base URL | `base_url` | 默认 `https://api.openai.com/v1/` |
| Model | `model` | 默认 `gpt-4.1-nano` |
| System Prompt | `system_prompt_{robotName}` | 每个机器人独立保存 |

支持三种 LLM 后端：**OpenAI** / **Ollama（本地）** / **Custom**。

---

## 六、联动关节（compoundMovements）

用于同一按键同时驱动多个关节（如手臂整体抬起）。

```ts
type CompoundMovement = {
  name: string
  keys: string[]           // 触发按键
  primaryJoint: number     // 主关节 servoId
  primaryFormula?: string  // 主关节增量公式（可用 primary, dependent 变量）
  dependents: {
    joint: number          // 从关节 servoId
    formula: string        // 从关节增量公式（可用 deltaPrimary, primary, dependent）
  }[]
}
```

**示例（so-arm100 手臂整体上下）：**
```ts
{
  name: "Jaw down & up",
  keys: ["8", "i"],
  primaryJoint: 2,          // Pitch 关节
  primaryFormula: "primary < 100 ? 1 : -1",
  dependents: [
    { joint: 3, formula: "primary < 100 ? -1.9 * deltaPrimary : 0.4 * deltaPrimary" },
    { joint: 4, formula: "primary < 100 ? 0.51 * deltaPrimary : -0.4 * deltaPrimary" },
  ]
}
```
