"""
Hand Wavelength  -  distance between your hands = wavelength of visible light.

  * Hands close together  -> violet (380 nm)
  * Hands far apart       -> red    (700 nm)
  * Hands touching        -> every color combines into white light
  * Live HUD shows wavelength (nm), distance %, hands detected, FPS, gesture

Controls:  q / ESC = quit      f = toggle fullscreen      h = toggle HUD
"""

import math
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Settings you can tweak
# --------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
WL_MIN, WL_MAX = 380.0, 700.0       # visible spectrum in nm
DIST_NEAR = 0.06                    # hand distance (fraction of frame width) = violet
DIST_FAR = 0.70                     # hand distance (fraction of frame width) = red
TOUCH_THRESHOLD = 0.07              # below this => white light mode
SMOOTHING = 0.25                    # 0..1, lower = smoother but laggier
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# Hand skeleton connections (MediaPipe 21-landmark layout)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGERTIPS = [4, 8, 12, 16, 20]
PALM_IDS = [0, 5, 9, 13, 17]


# --------------------------------------------------------------------------
# Physics / color helpers
# --------------------------------------------------------------------------
def wavelength_to_rgb(wl):
    """Visible wavelength (380-780 nm) -> (r, g, b) floats 0..1  (Dan Bruton)."""
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif 580 <= wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    elif 645 <= wl <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r = g = b = 0.0
    # intensity falloff near the edges of vision
    if 380 <= wl < 420:
        f = 0.3 + 0.7 * (wl - 380) / 40
    elif 420 <= wl < 701:
        f = 1.0
    elif 701 <= wl <= 780:
        f = 0.3 + 0.7 * (780 - wl) / 80
    else:
        f = 0.0
    # Pure violet is very dark on screen; keep it punchy for the visual.
    f = max(f, 0.75)
    return tuple(min(1.0, c * f) for c in (r, g, b))


def rgb_bgr255(rgb):
    return int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)


def distance_to_t(d_norm):
    """Normalized hand distance -> 0 (near) .. 1 (far)."""
    return float(np.clip((d_norm - DIST_NEAR) / (DIST_FAR - DIST_NEAR), 0, 1))


def classify_gesture(lm_px):
    """Very small gesture classifier from 21 pixel landmarks."""
    palm = np.mean([lm_px[i] for i in PALM_IDS], axis=0)
    scale = np.linalg.norm(np.array(lm_px[0]) - np.array(lm_px[9])) + 1e-6
    tips = np.mean([np.linalg.norm(np.array(lm_px[i]) - palm) for i in FINGERTIPS])
    ratio = tips / scale
    if ratio < 0.85:
        return "Fist", ratio
    if ratio > 1.6:
        return "Open", ratio
    return "Relaxed", ratio


# --------------------------------------------------------------------------
# Hand tracker (works with old `mp.solutions` AND new Tasks API)
# --------------------------------------------------------------------------
class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=2, model_complexity=0,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks.python import vision, BaseOptions
            self._ensure_model()
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_tracking_confidence=0.5)
            self.hands = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ts = -1

    @staticmethod
    def _ensure_model():
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
            return
        print("Downloading hand model (one time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    def process(self, frame_bgr):
        """Returns list of hands; each hand = list of 21 (x, y) pixel tuples."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = []
        if self.legacy:
            res = self.hands.process(rgb)
            for hl in (res.multi_hand_landmarks or []):
                out.append([(p.x * w, p.y * h) for p in hl.landmark])
        else:
            ts = int((time.monotonic() - self._t0) * 1000)
            ts = max(ts, self._last_ts + 1)          # must strictly increase
            self._last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.hands.detect_for_video(img, ts)
            for hl in res.hand_landmarks:
                out.append([(p.x * w, p.y * h) for p in hl])
        return out


# --------------------------------------------------------------------------
# Particles
# --------------------------------------------------------------------------
class Particles:
    def __init__(self, cap=350):
        self.p = []      # [x, y, vx, vy, life, color]
        self.cap = cap

    def emit(self, x, y, color, n=3):
        for _ in range(n):
            if len(self.p) >= self.cap:
                break
            a = np.random.uniform(0, 2 * math.pi)
            s = np.random.uniform(0.5, 3.0)
            self.p.append([x, y, math.cos(a) * s, math.sin(a) * s - 0.6,
                           1.0, color])

    def step_and_draw(self, layer):
        alive = []
        for q in self.p:
            q[0] += q[2]
            q[1] += q[3]
            q[4] -= 0.025
            if q[4] > 0:
                c = tuple(int(v * q[4]) for v in q[5])
                cv2.circle(layer, (int(q[0]), int(q[1])), 2, c, -1, cv2.LINE_AA)
                alive.append(q)
        self.p = alive


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def draw_hand(layer, lm, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(layer, _pt(lm[a]), _pt(lm[b]), color, 3, cv2.LINE_AA)
    for p in lm:
        cv2.circle(layer, _pt(p), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(layer, _pt(p), 7, color, 2, cv2.LINE_AA)


def _pt(p):
    return int(p[0]), int(p[1])


def draw_beams(layer, hand_a, hand_b, color):
    """Light beams between matching fingertips + palms of both hands."""
    for i in FINGERTIPS + [0, 9]:
        cv2.line(layer, _pt(hand_a[i]), _pt(hand_b[i]), color, 2, cv2.LINE_AA)


def panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w]
    dark = np.full_like(sub, (28, 26, 30))
    img[y:y + h, x:x + w] = cv2.addWeighted(dark, alpha, sub, 1 - alpha, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 78, 85), 1, cv2.LINE_AA)


def hud_row(img, x, y, label, value, vcolor=(200, 255, 120)):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (x, y), f, 0.5, (210, 210, 215), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(value, f, 0.55, 2)
    cv2.putText(img, value, (x + 262 - tw, y), f, 0.55, vcolor, 2, cv2.LINE_AA)


def draw_hud(img, n_hands, fps, gesture, spread, wl, dist_pct, white):
    panel(img, 16, 16, 290, 62)
    hud_row(img, 28, 40, "Hands Detected:", str(n_hands))
    hud_row(img, 28, 64, "FPS:", f"{fps:.0f}")
    panel(img, 16, 88, 290, 62)
    hud_row(img, 28, 112, "Gesture:", gesture)
    hud_row(img, 28, 136, "Spread:", f"{spread:.0f}%")
    panel(img, 16, 160, 290, 86)
    # lighten the color a bit so dark blues/violets stay readable on the panel
    base = np.array(rgb_bgr255(wavelength_to_rgb(wl)), dtype=float)
    col = tuple(int(v) for v in (base * 0.6 + 255 * 0.4))
    if white:
        col = (255, 255, 255)
    hud_row(img, 28, 184, "Wavelength:", "WHITE" if white else f"{wl:.0f} nm", col)
    hud_row(img, 28, 208, "Spectrum Range:", "380-700nm", (170, 170, 175))
    hud_row(img, 28, 232, "Distance:", f"{dist_pct:.0f}%")


def draw_spectrum_bar(img, t, white):
    """Bottom bar showing the visible spectrum with a marker at current value."""
    h, w = img.shape[:2]
    bw, bh = int(w * 0.5), 14
    x0, y0 = (w - bw) // 2, h - 44
    for i in range(bw):
        wl = WL_MIN + (WL_MAX - WL_MIN) * i / bw
        img[y0:y0 + bh, x0 + i] = rgb_bgr255(wavelength_to_rgb(wl))
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (230, 230, 230), 1)
    mx = x0 + int(t * bw)
    cv2.line(img, (mx, y0 - 8), (mx, y0 + bh + 8), (255, 255, 255), 2, cv2.LINE_AA)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "380nm", (x0 - 6, y0 + bh + 24), f, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "700nm", (x0 + bw - 36, y0 + bh + 24), f, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def render(frame, hands, state, particles, fps, show_hud=True):
    """Draw everything onto `frame` (in place) and return it.

    `state` is a dict that persists across frames (smoothed t, white level).
    """
    h, w = frame.shape[:2]
    frame[:] = cv2.addWeighted(frame, 0.78, np.zeros_like(frame), 0.22, 0)  # dim bg

    n = len(hands)
    gesture, spread = "-", 0.0
    if n:
        gesture, ratio = classify_gesture(hands[0])
        spread = float(np.clip((ratio - 0.6) / (2.0 - 0.6), 0, 1) * 100)

    # --- distance -> wavelength ------------------------------------------
    target_t = state["t"]
    d_norm = None
    if n == 2:
        ca = np.mean([hands[0][i] for i in PALM_IDS], axis=0)
        cb = np.mean([hands[1][i] for i in PALM_IDS], axis=0)
        d_norm = float(np.linalg.norm(ca - cb)) / w
        target_t = distance_to_t(d_norm)
    elif n == 1:
        # one hand: use thumb-to-pinky spread so it still does something
        d_norm = abs(hands[0][4][0] - hands[0][20][0]) / w * 3
        target_t = distance_to_t(d_norm)
    state["t"] += (target_t - state["t"]) * SMOOTHING
    t = state["t"]
    wl = WL_MIN + t * (WL_MAX - WL_MIN)
    color = rgb_bgr255(wavelength_to_rgb(wl))

    touching = n == 2 and d_norm is not None and d_norm < TOUCH_THRESHOLD
    # white level eases in/out
    state["white"] += ((1.0 if touching else 0.0) - state["white"]) * 0.18
    white = state["white"]

    # --- glow layer ----------------------------------------------------
    layer = np.zeros_like(frame)
    for hand in hands:
        draw_hand(layer, hand, color)
        for i in FINGERTIPS:
            particles.emit(hand[i][0], hand[i][1], color, 1)
    if n == 2:
        draw_beams(layer, hands[0], hands[1], color)
        mid = ((ca + cb) / 2).astype(int)
        label = f"{wl:.0f} nm"
        f = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, f, 0.8, 2)
        cv2.rectangle(layer, (mid[0] - tw // 2 - 10, mid[1] - th - 10),
                      (mid[0] + tw // 2 + 10, mid[1] + 10), color, -1)
    particles.step_and_draw(layer)

    blur_big = cv2.GaussianBlur(layer, (0, 0), 18)
    blur_small = cv2.GaussianBlur(layer, (0, 0), 6)
    frame[:] = cv2.add(frame, cv2.add(blur_big, blur_small))
    frame[:] = cv2.add(frame, layer)

    if n == 2:  # crisp label text on top
        f = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, f, 0.8, 2)
        cv2.putText(frame, label, (mid[0] - tw // 2, mid[1] + 2), f, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # --- white light flash when hands touch -----------------------------
    if white > 0.01:
        frame[:] = cv2.addWeighted(frame, 1 - white, np.full_like(frame, 255), white, 0)
        if white > 0.5:
            f = cv2.FONT_HERSHEY_SIMPLEX
            msg = "ALL COLORS = WHITE LIGHT"
            (tw, _), _ = cv2.getTextSize(msg, f, 1.0, 2)
            cv2.putText(frame, msg, ((w - tw) // 2, h // 2), f, 1.0, (60, 60, 60), 2, cv2.LINE_AA)

    if show_hud:
        draw_hud(frame, n, fps, gesture, spread, wl,
                 (d_norm / DIST_FAR * 100) if d_norm is not None else 0, white > 0.5)
        draw_spectrum_bar(frame, t, white > 0.5)
    return frame


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAM_INDEX}. Try changing CAM_INDEX.")

    tracker = HandTracker()
    particles = Particles()
    state = {"t": 0.5, "white": 0.0}
    times = deque(maxlen=30)
    show_hud, fullscreen = True, False
    win = "Hand Wavelength"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)            # mirror like a selfie
        hands = tracker.process(frame)

        now = time.perf_counter()
        times.append(now)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

        render(frame, hands, state, particles, fps, show_hud)
        cv2.imshow(win, frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("h"):
            show_hud = not show_hud
        if k == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
