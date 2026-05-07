"""
Star Hand — your hand rendered as a glowing constellation.
Anchored at a base platform, the arm extends upward mirroring your pose.

Controls:
  Q / Esc  — quit
  +/-      — zoom
  S        — screenshot
"""

import cv2
import numpy as np
import argparse
import time
import math
import os
from collections import deque

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks import python as mp_python
from mediapipe import ImageFormat

# ---------------------------------------------------------------------------
# Landmark topology
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_C, THUMB_M, THUMB_I, THUMB_T = 1, 2, 3, 4
INDEX_M, INDEX_P, INDEX_D, INDEX_T = 5, 6, 7, 8
MID_M, MID_P, MID_D, MID_T = 9, 10, 11, 12
RING_M, RING_P, RING_D, RING_T = 13, 14, 15, 16
PINK_M, PINK_P, PINK_D, PINK_T = 17, 18, 19, 20

# Constellation threads (bone connections)
THREADS = [
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm mesh
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17),
    (2, 5),
    # Vertical "cable" from base to wrist
    (-1, 0),  # -1 = base anchor (special)
]

# Which finger each landmark belongs to (0..4), None = wrist
LANDMARK_FINGER = [None] * 21
for fi, grp in enumerate([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                           (13, 14, 15, 16), (17, 18, 19, 20)]):
    for idx in grp:
        LANDMARK_FINGER[idx] = fi

FINGER_COLORS_BGR = [
    (255, 170, 48),   # thumb  — amber
    (255, 216, 64),   # index  — ice cyan
    (80, 255, 80),    # middle — neon green
    (192, 96, 255),   # ring   — hot pink
    (255, 128, 192),  # pinky  — lavender
]
WRIST_COLOR = (255, 200, 180)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "hand_landmarker.task")

# ---------------------------------------------------------------------------
# Star constellation rendering (vertical, bottom-anchored)
# ---------------------------------------------------------------------------

def draw_constellation(canvas, img_landmarks, wrist_x, wrist_y, hand_scale,
                       panel_x, panel_y, panel_w, panel_h, time_sec):
    """Render the hand as white points/lines on pure black.

    Base platform at the bottom, arm extends upward.
    Monochrome white-on-black minimalist aesthetic.
    """

    # Pure black background
    canvas[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = (0, 0, 0)

    if img_landmarks is None:
        return

    lms = img_landmarks
    n = len(lms)

    # ---- Compute screen positions ----
    pts = np.zeros((n + 1, 2), dtype=np.float32)  # +1 for base anchor

    wrist_lm_y = lms[WRIST].y
    for i in range(n):
        pts[i, 0] = panel_x + panel_w // 2 + (lms[i].x - 0.5) * hand_scale
        # Subtract delta so fingertips go UP (lower y) on screen.
        # When hand moves up (wrist_lm_y decreases), all points shift up.
        pts[i, 1] = wrist_y - (wrist_lm_y - lms[i].y) * hand_scale * 0.75

    # Base anchor (index -1) at bottom center
    base_x = panel_x + panel_w // 2
    base_y = panel_y + panel_h - 30
    pts[n, 0] = base_x
    pts[n, 1] = base_y

    bx, by = int(base_x), int(base_y)

    # ---- Base platform (white outline ellipse) ----
    bw = int(panel_w * 0.22)
    cv2.ellipse(canvas, (bx, by), (bw, bw // 3), 0, 180, 360,
                (80, 80, 80), 1, cv2.LINE_AA)
    # Inner ellipse
    cv2.ellipse(canvas, (bx, by), (bw - 25, bw // 3 - 8), 0, 180, 360,
                (50, 50, 50), 1, cv2.LINE_AA)

    # ---- Base pillar (white wireframe) ----
    pillar_w = 14
    pillar_top = by - 25
    cv2.line(canvas, (bx - pillar_w, by), (bx - 8, pillar_top), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(canvas, (bx + pillar_w, by), (bx + 8, pillar_top), (100, 100, 100), 1, cv2.LINE_AA)

    # ---- Cable from base to wrist (white line with glow) ----
    wrist_sx = pts[WRIST, 0]
    wrist_sy = pts[WRIST, 1]
    # Outer glow
    cv2.line(canvas, (bx, pillar_top), (int(wrist_sx), int(wrist_sy)),
             (30, 30, 30), 4, cv2.LINE_AA)
    # Core line
    cv2.line(canvas, (bx, pillar_top), (int(wrist_sx), int(wrist_sy)),
             (180, 180, 180), 1, cv2.LINE_AA)

    # ---- Draw connecting threads (white lines) ----
    for a, b in THREADS:
        if a == -1 or a >= n or b >= n:
            continue
        p1 = (int(pts[a, 0]), int(pts[a, 1]))
        p2 = (int(pts[b, 0]), int(pts[b, 1]))

        # Glow layer
        cv2.line(canvas, p1, p2, (25, 25, 25), 3, cv2.LINE_AA)
        # Core line
        cv2.line(canvas, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    # ---- Draw node points ----
    for i in range(n):
        px, py = int(pts[i, 0]), int(pts[i, 1])

        if not (panel_x - 20 < px < panel_x + panel_w + 20 and
                panel_y - 20 < py < panel_y + panel_h + 20):
            continue

        is_tip = i in (THUMB_T, INDEX_T, MID_T, RING_T, PINK_T)
        is_joint = i in (THUMB_M, THUMB_I, INDEX_P, INDEX_D,
                         MID_P, MID_D, RING_P, RING_D, PINK_P, PINK_D)

        # Determine point size based on type
        if is_tip:
            outer_r, mid_r, core_r = 10, 5, 3
        elif i == WRIST:
            outer_r, mid_r, core_r = 14, 8, 4
        elif is_joint:
            outer_r, mid_r, core_r = 6, 3, 2
        else:
            outer_r, mid_r, core_r = 4, 2, 1

        pulse = 0.7 + 0.3 * math.sin(time_sec * 2.2 + i * 0.55)

        # Outer glow
        cv2.circle(canvas, (px, py), int(outer_r * pulse), (25, 25, 25), -1, cv2.LINE_AA)
        # Mid
        cv2.circle(canvas, (px, py), int(mid_r * pulse), (100, 100, 100), -1, cv2.LINE_AA)
        # Core
        cv2.circle(canvas, (px, py), int(core_r * pulse), (230, 230, 230), -1, cv2.LINE_AA)
        # Specular
        if core_r >= 2:
            cv2.circle(canvas, (px, py), max(1, core_r // 2), (255, 255, 255), -1, cv2.LINE_AA)

    # ---- Wrist hub (larger, dimmer) ----
    wx, wy = int(wrist_sx), int(wrist_sy)
    cv2.circle(canvas, (wx, wy), 16, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(canvas, (wx, wy), 10, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.circle(canvas, (wx, wy), 5, (160, 160, 160), -1, cv2.LINE_AA)
    cv2.circle(canvas, (wx, wy), 2, (240, 240, 240), -1, cv2.LINE_AA)


def draw_overlay(canvas, fps, gesture_name, zoom_pct,
                 panel_x, panel_y, panel_w, panel_h):
    """Status bar on the constellation panel."""
    # Top bar
    cv2.rectangle(canvas, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + 34), (0, 0, 0), -1)
    cv2.putText(canvas, "STAR HAND :: CONSTELLATION",
                (panel_x + 14, panel_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 160, 200), 1, cv2.LINE_AA)
    if gesture_name:
        cv2.putText(canvas, gesture_name,
                    (panel_x + panel_w - 140, panel_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

    # Bottom bar
    yb = panel_y + panel_h - 28
    cv2.rectangle(canvas, (panel_x, yb), (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)
    cv2.putText(canvas, f"FPS: {fps:.0f}  |  Zoom: {zoom_pct}%  |  +/- zoom  S screenshot  Q quit",
                (panel_x + 14, yb + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 120, 160), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Gesture recognition
# ---------------------------------------------------------------------------

def recognise_gesture(landmarks) -> str:
    tip = lambda i: landmarks[[THUMB_T, INDEX_T, MID_T, RING_T, PINK_T][i]]
    pip = lambda i: landmarks[[THUMB_I, INDEX_P, MID_P, RING_P, PINK_P][i]]

    def ext(i, thumb=False):
        if thumb:
            return abs(tip(i).x - landmarks[INDEX_M].x) > 0.06
        return tip(i).y < pip(i).y - 0.015

    e = [ext(0, True), ext(1), ext(2), ext(3), ext(4)]
    t, idx, mid, rng, pnk = e

    if all(e):                return "Open Palm"
    if not any(e):            return "Fist"
    if t and not any(e[1:]): return "Thumbs Up"
    if idx and mid and not t and not rng and not pnk: return "Peace"
    if idx and pnk and not mid and not rng: return "Rock On"
    if idx and not any(e[1:]) and not t:     return "Point"
    return ""


# ---------------------------------------------------------------------------
# Camera panel
# ---------------------------------------------------------------------------

def draw_camera_panel(canvas, frame, img_landmarks_list,
                      panel_x, panel_y, panel_w, panel_h):
    h, w = frame.shape[:2]
    cam = cv2.resize(frame, (panel_w, panel_h))
    canvas[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = cam

    if img_landmarks_list:
        for landmarks in img_landmarks_list:
            for a, b in THREADS:
                if a < 0 or b < 0 or a >= len(landmarks) or b >= len(landmarks):
                    continue
                ax = panel_x + int(landmarks[a].x * panel_w)
                ay = panel_y + int(landmarks[a].y * panel_h)
                bx = panel_x + int(landmarks[b].x * panel_w)
                by = panel_y + int(landmarks[b].y * panel_h)
                cv2.line(canvas, (ax, ay), (bx, by), (200, 200, 200), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + 30), (0, 0, 0), -1)
    cv2.putText(canvas, "CAMERA FEED", (panel_x + 12, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 240, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Star Hand")
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0 if args.input is None else args.input)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.7, min_tracking_confidence=0.6,
        running_mode=VisionTaskRunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(hand_options)

    # Zoom control: hand_scale = base_scale * zoom_mult
    zoom_mult = 1.0
    base_scale = 1100.0  # maps normalized landmark coords → pixels
    fps_hist = deque(maxlen=20)
    prev_tick = time.perf_counter()
    gesture_name = ""

    win_name = "Star Hand"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 60)
    print("  STAR HAND — Constellation Visualization")
    print("  Your hand becomes a glowing star field")
    print("=" * 60)
    print("  +/-  zoom    S  screenshot    Q  quit")
    print("=" * 60)

    screenshot_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_flip and args.input is None:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(ImageFormat.SRGB, rgb)
        result = detector.detect(mp_img)

        # Layout: camera (left) | constellation (right)
        canvas_w, canvas_h = 1500, 850
        cam_w = int(canvas_w * 0.42)
        star_w = canvas_w - cam_w

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (10, 12, 20)

        t_now = time.perf_counter()

        # Camera panel
        draw_camera_panel(canvas, frame,
                          result.hand_landmarks if result.hand_landmarks else [],
                          0, 0, cam_w, canvas_h)

        # Constellation panel
        gesture_name = ""
        hand_scale = base_scale * zoom_mult

        if result.hand_landmarks:
            ilm = result.hand_landmarks[0]
            gesture_name = recognise_gesture(ilm)

            # Wrist anchor: ~15% from bottom of panel
            wrist_anchor_x = cam_w + star_w // 2
            # Anchor tracks wrist position: centered when wrist at y≈0.6,
            # moves up when hand rises, down when hand lowers
            anchor_base = canvas_h * 0.55
            anchor_range = canvas_h * 0.3
            wrist_anchor_y = int(anchor_base + (ilm[WRIST].y - 0.6) * anchor_range)

            draw_constellation(canvas, ilm, wrist_anchor_x, wrist_anchor_y,
                               hand_scale, cam_w, 0, star_w, canvas_h, t_now)
            draw_overlay(canvas, np.mean(fps_hist) if fps_hist else 0,
                         gesture_name, int(zoom_mult * 100),
                         cam_w, 0, star_w, canvas_h)
        else:
            # Empty state
            cv2.rectangle(canvas, (cam_w, 0), (cam_w + star_w, canvas_h), (10, 12, 20), -1)
            cv2.putText(canvas, "SHOW YOUR HAND", (cam_w + star_w // 2 - 145, canvas_h // 2 - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (60, 75, 110), 1, cv2.LINE_AA)
            scan_y = int(canvas_h * 0.5 + 15 * math.sin(t_now * 2))
            cv2.line(canvas, (cam_w + 100, scan_y), (cam_w + star_w - 100, scan_y),
                     (30, 45, 70), 1, cv2.LINE_AA)
            draw_overlay(canvas, np.mean(fps_hist) if fps_hist else 0,
                         "", int(zoom_mult * 100), cam_w, 0, star_w, canvas_h)

        # Global title bar
        cv2.rectangle(canvas, (0, 0), (canvas_w, 24), (6, 8, 16), -1)
        cv2.putText(canvas, "STAR HAND :: Constellation  |  MediaPipe + OpenCV",
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 120, 170), 1, cv2.LINE_AA)

        # FPS
        fps_hist.append(1.0 / max(t_now - prev_tick, 0.001))
        prev_tick = t_now

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            screenshot_n += 1
            fn = f"star_hand_{screenshot_n:03d}.png"
            cv2.imwrite(fn, canvas)
            print(f"Screenshot: {fn}")
        elif key in (ord("="), ord("+")):
            zoom_mult = min(3.0, zoom_mult + 0.1)
        elif key == ord("-"):
            zoom_mult = max(0.3, zoom_mult - 0.1)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Star Hand shut down.")


if __name__ == "__main__":
    main()
