# Hand Gesture Lab — 项目总结

## 项目概述

基于 MediaPipe + OpenCV 的实时手势交互实验，包含两个子项目：

| 项目 | 路径 | 功能 |
|------|------|------|
| Gesture Lab | `gesture_lab.py` | 识别 9 种手势并标注 |
| Star Hand | `robot_arm_mimic/` | 手部星座点阵实时可视化 |

## 技术栈

- **Python 3.11**
- **MediaPipe 0.10.35** — HandLandmarker 模型，提取 21 个手部关键点
- **OpenCV 4.13** — 视频捕获与画面渲染
- **NumPy** — 3D 坐标变换与投影

## Gesture Lab

识别 9 种手势：Fist、Open Palm、Thumbs Up、Peace、OK、Point、Rock On、Call Me、Pinch。

手势判定基于指尖关节相对位置：
- **拇指**：与食指 MCP 的水平距离
- **其余四指**：指尖 Y 坐标是否高于 PIP 关节

界面包含手势名称、FPS、底部手势历史时间线。

## Star Hand

### 视觉设计

- 纯黑背景 + 白色点线渲染
- 底座在画面下方，手臂竖直向上延伸
- 21 个手部关键点 → 白色光点（外层辉光 → 中层 → 核心 → 高光）
- 骨骼连接 → 白色星座线
- 指尖比关节更大更亮，手腕是最大锚点

### 坐标映射

以手腕为锚点，图像坐标映射到画面坐标：

```
screen_x = center_x + (lm.x - 0.5) * scale
screen_y = wrist_anchor_y - (wrist_lm.y - lm.y) * scale
```

锚点动态跟随手腕在画面中的位置，手举高则星座上移，手放低则下移。

### 交互控制

| 按键 | 功能 |
|------|------|
| +/- | 缩放 |
| S | 截图 |
| Q | 退出 |

## 踩坑记录

1. **MediaPipe 0.10 API 变更**：`mp.solutions` → `mp.tasks`，需要下载 7.5MB 的 `hand_landmarker.task` 模型文件
2. **OpenCV 4.13 颜色类型**：numpy int 无法直接用作颜色参数，需显式 `int()` 转换
3. **Y 轴方向**：屏幕坐标 y=0 在顶部，而手部关键点 y=0 也在图像顶部，映射时需正确处理符号
4. **GitHub 推送**：macOS Keychain 凭据管理器已存储凭据，`git push` 无需额外配置

## 运行方式

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 手势识别
python gesture_lab.py

# 星空点阵
python robot_arm_mimic/robot_arm_mimic.py
```

## 仓库

https://github.com/Wenshen-Liu/hand-gesture-lab
