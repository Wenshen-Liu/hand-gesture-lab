# Hand Gesture Lab

Two real-time hand-tracking experiments using MediaPipe + OpenCV.

## Projects

### 1. Gesture Lab (`gesture_lab.py`)

Recognises 9 hand gestures from webcam or video:

| Gesture | Trigger |
|---------|---------|
| ✊ Fist | All fingers curled |
| ✋ Open Palm | All fingers extended |
| 👍 Thumbs Up | Thumb up, others curled |
| ✌️ Peace | Index + middle up |
| 👌 OK | Thumb-index pinch, others extended |
| ☝️ Point | Only index extended |
| 🤘 Rock On | Index + pinky up |
| 🤙 Call Me | Thumb + pinky up |
| 🤏 Pinch | Thumb-index pinch |

```bash
python gesture_lab.py                     # webcam
python gesture_lab.py --input video.mp4   # file
python gesture_lab.py --max-hands 2       # two hands
```

### 2. Star Hand (`robot_arm_mimic/`)

Renders your hand as a white point-cloud constellation on a pure black background. A base platform anchors the bottom; the constellation extends upward, mirroring your pose in real-time.

```bash
python robot_arm_mimic/robot_arm_mimic.py
```

| Key | Action |
|-----|--------|
| +/- | Zoom |
| S   | Screenshot |
| Q   | Quit |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a working webcam.

