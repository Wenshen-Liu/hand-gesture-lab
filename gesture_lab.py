"""
Gesture Lab — Real-time hand gesture recognition from video.
Supports webcam and pre-recorded video files.

Recognized gestures:
  ✊ Fist, ✋ Open Palm, 👍 Thumbs Up, ✌️ Peace, 👌 OK,
  ☝️ Point, 🤘 Rock On, 🤙 Call Me, 🤏 Pinch
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe import ImageFormat

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
WRIST = 0
THUMB =  (1, 2, 3, 4)     # CMC, MCP, IP, TIP
INDEX =  (5, 6, 7, 8)     # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING =   (13, 14, 15, 16)
PINKY =  (17, 18, 19, 20)

FINGER_TIPS = [THUMB[3], INDEX[3], MIDDLE[3], RING[3], PINKY[3]]
FINGER_PIPS = [THUMB[2], INDEX[1], MIDDLE[1], RING[1], PINKY[1]]
FINGER_MCPS = [THUMB[1], INDEX[0], MIDDLE[0], RING[0], PINKY[0]]

MODEL_PATH = __file__.rsplit("/", 1)[0] + "/hand_landmarker.task"

# ---------------------------------------------------------------------------
# Gesture detection helpers
# ---------------------------------------------------------------------------

def is_finger_extended(lm, finger_mcp, finger_pip, finger_tip, thumb=False):
    if thumb:
        return abs(lm[finger_tip].x - lm[INDEX[0]].x) > 0.08
    return lm[finger_tip].y < lm[finger_pip].y - 0.02


def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def recognise(lm) -> str:
    extended = [
        is_finger_extended(lm, FINGER_MCPS[0], FINGER_PIPS[0], FINGER_TIPS[0], thumb=True),
        is_finger_extended(lm, FINGER_MCPS[1], FINGER_PIPS[1], FINGER_TIPS[1]),
        is_finger_extended(lm, FINGER_MCPS[2], FINGER_PIPS[2], FINGER_TIPS[2]),
        is_finger_extended(lm, FINGER_MCPS[3], FINGER_PIPS[3], FINGER_TIPS[3]),
        is_finger_extended(lm, FINGER_MCPS[4], FINGER_PIPS[4], FINGER_TIPS[4]),
    ]
    thumb, index, middle, ring, pinky = extended

    tip = lambda i: lm[FINGER_TIPS[i]]
    pip = lambda i: lm[FINGER_PIPS[i]]
    mcp = lambda i: lm[FINGER_MCPS[i]]

    # 👍 Thumbs Up — thumb up, others folded
    if thumb and not any([index, middle, ring, pinky]):
        if tip(0).y < mcp(0).y - 0.03:
            return "Thumbs Up"

    # ✌️ Peace — index + middle up, thumb + ring + pinky down
    if index and middle and not thumb and not ring and not pinky:
        if distance(tip(1), tip(2)) > 0.04:
            return "Peace"

    # 🤘 Rock On — index + pinky up, others down
    if index and pinky and not middle and not ring:
        return "Rock On"

    # ☝️ Point — only index up
    if index and not any([middle, ring, pinky]) and not thumb:
        return "Point"

    # 🤙 Call Me — thumb + pinky up, others down
    if thumb and pinky and not any([index, middle, ring]):
        return "Call Me"

    # 👌 OK — thumb and index tips close, other 3 extended
    if middle and ring and pinky and not index:
        if distance(tip(0), tip(1)) < 0.05:
            return "OK"

    # 🤏 Pinch — thumb and index tips close, others extended
    if distance(tip(0), tip(1)) < 0.04 and middle and ring and pinky:
        return "Pinch"

    # ✋ Open — all 5 extended
    if all(extended):
        return "Open Palm"

    # ✊ Fist — none extended
    if not any(extended):
        return "Fist"

    return "Unknown"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_ui(image, gesture, fps, gesture_history, hand_present):
    h, w = image.shape[:2]

    # semi-transparent top bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # gesture label
    color = (0, 255, 120) if hand_present else (100, 100, 120)
    label = gesture if hand_present else "No Hand Detected"
    cv2.putText(image, label, (24, 58), cv2.FONT_HERSHEY_DUPLEX,
                1.5, color, 2, cv2.LINE_AA)

    # FPS
    cv2.putText(image, f"FPS: {fps:.0f}", (w - 130, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 220), 1, cv2.LINE_AA)

    # gesture history timeline at bottom
    if gesture_history:
        bar_h = 60
        y0 = h - bar_h - 10
        cv2.rectangle(image, (10, y0), (w - 10, h - 10), (30, 30, 50), -1)
        n = len(gesture_history)
        bar_w = (w - 30) // n
        for i, g in enumerate(gesture_history):
            x = 15 + i * bar_w
            hue = hash(g) % 180
            c = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(image, (x, y0 + 5), (x + bar_w - 4, h - 15), c, -1)
            cv2.putText(image, g[:4], (x + 2, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return image


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gesture Lab — real-time hand gesture recognition")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Video file path (default: webcam)")
    parser.add_argument("--flip", action="store_true", default=True,
                        help="Flip camera horizontally (mirror mode)")
    parser.add_argument("--no-flip", dest="flip", action="store_false")
    parser.add_argument("--max-hands", type=int, default=1)
    args = parser.parse_args()

    cap = cv2.VideoCapture(0 if args.input is None else args.input)
    if not cap.isOpened():
        print(f"Error: cannot open {'webcam' if args.input is None else args.input}")
        return

    # MediaPipe new API (0.10.x)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        running_mode=mp_python.vision.RunningMode.IMAGE,
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    gesture_history = deque(maxlen=30)
    fps_history = deque(maxlen=15)
    prev_tick = time.perf_counter()

    print("Gesture Lab running. Press Q to quit.")
    print("Gestures: Fist | Open Palm | Thumbs Up | Peace | OK | Point | Rock On | Call Me | Pinch")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip and args.input is None:
            frame = cv2.flip(frame, 1)

        # Convert BGR → RGB → MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(ImageFormat.SRGB, rgb)
        result = hand_detector.detect(mp_img)

        gesture = ""
        hand_present = False

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                gesture = recognise(hand_landmarks)
                hand_present = True
                vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=vision.drawing_utils.DrawingSpec(
                        color=(0, 220, 120), thickness=2, circle_radius=3),
                    connection_drawing_spec=vision.drawing_utils.DrawingSpec(
                        color=(80, 180, 255), thickness=2),
                )

        now = time.perf_counter()
        fps_history.append(1.0 / max(now - prev_tick, 0.001))
        prev_tick = now
        fps = np.mean(fps_history)

        gesture_history.append(gesture)
        frame = draw_ui(frame, gesture, fps, gesture_history, hand_present)

        cv2.imshow("Gesture Lab", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
