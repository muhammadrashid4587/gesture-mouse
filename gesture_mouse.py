"""gesture-mouse: drive an on-screen board with both hands, seen through the webcam.

Pinch (thumb + index) to press buttons, grab cards and move the slider. Let go of a card
while your hand is moving to throw it.
Make a fist to freeze that hand: it lets go and ignores the board until it opens. Press q to quit.

    uv run python gesture_mouse.py
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
WINDOW = "gesture-mouse"
W, H = 1280, 720

# Direct grabbing: each hand's cursor is the point between its thumb tip and index tip.
POINTER_SMOOTHING = 1.0  # One Euro min cutoff (Hz): lower = steadier when still, more lag
POINTER_RESPONSE = 0.03  # One Euro beta: higher = less lag when moving fast
PREDICT = 0.04          # seconds the drawn cursor looks ahead along the hand's motion (hides camera lag)

# Gesture thresholds, fitted to recorded pinches, clicks and fists.
PINCH_ON = 0.22         # 2D thumb-index gap / palm length to start a pinch
PINCH_OFF = 0.30        # ... and to end it (quick clicks only open to ~0.35)
HOLD_OFF = 0.42         # ... but while holding something (a held pinch loosens to ~0.4 as you move)
DEBOUNCE = 2            # frames a pinch change must hold before it counts
INDEX_CURLED = 0.80     # index tip-to-wrist / pip-to-wrist below this: index folded (fist)
OTHERS_CURLED = 0.90    # same for middle, ring and pinky
INDEX_OPEN = 0.90       # index straightened past this: the fist has opened
FIST_THUMB_GAP = 0.15   # in a fist the thumb rests beside the index, not on its tip
FIST_FRAMES = 6         # frames a fist must hold before the hand freezes
LOST_GRACE = 0.35       # seconds a hand may vanish before it drops what it holds

# Throwing: let go of a card while moving and it slides on, slows down and bounces off the edges.
THROW_MIN = 350         # px/s: let go slower than this and the card just drops
THROW_MAX = 5000        # px/s
FRICTION = 2.5          # 1/s: how fast a thrown card slows down (it slides ~speed / FRICTION px)
BOUNCE = 0.6            # share of its speed a card keeps when it hits an edge

LEFT_COLOR = (0, 150, 255)      # orange (BGR)
RIGHT_COLOR = (255, 170, 40)    # blue
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip) of index..pinky
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
BUTTON_COLORS = [(110, 110, 110), (70, 70, 225), (70, 175, 70), (215, 120, 40),
                 (40, 195, 230), (190, 80, 190)]


# --------------------------------------------------------------------------- drawing

def text(img, s, org, scale=0.7, color=WHITE, thick=2, align="left", outline=True):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if align == "center":
        x, y = x - tw // 2, y + th // 2
    elif align == "right":
        x -= tw
    if outline:
        # Offset copies, not a thicker stroke: in OpenCV 5 thickness also widens the spacing.
        r = 2 if scale >= 0.6 else 1
        for dx, dy in ((-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (r, r), (-r, r), (r, -r)):
            cv2.putText(img, s, (x + dx, y + dy), FONT, scale, BLACK, thick, AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, AA)


def rounded_rect(img, x0, y0, x1, y1, r, color, thickness=-1):
    x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(r)
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
        return
    t = thickness
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, t, AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, t, AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, t, AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, t, AA)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, t, AA)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, t, AA)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, t, AA)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, t, AA)


def lighten(color, k=0.5):
    return tuple(int(c + (255 - c) * k) for c in color)


# --------------------------------------------------------------------------- board

class Button:
    def __init__(self, name, rect):
        self.name, self.rect = name, rect
        self.count = 0
        self.flash = -1.0

    def hit(self, p):
        x0, y0, x1, y1 = self.rect
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


class Card:
    W, H = 230, 150

    def __init__(self, name, x, y, color):
        self.name, self.color = name, color
        self.pos = np.array([x, y], dtype=float)
        self.held_by = None
        self.offset = np.zeros(2)
        self.vel = np.zeros(2)  # px/s while sliding after a throw

    def hit(self, p):
        x, y = self.pos
        return x <= p[0] <= x + self.W and y <= p[1] <= y + self.H


class Slider:
    name = "slider"
    KNOB = 28

    def __init__(self, x0, x1, y):
        self.x0, self.x1, self.y = x0, x1, y
        self.value = 0.5
        self.held_by = None
        self.offset = 0.0

    @property
    def knob_x(self):
        return self.x0 + self.value * (self.x1 - self.x0)

    def hit(self, p):
        return np.hypot(p[0] - self.knob_x, p[1] - self.y) <= self.KNOB * 1.6


class Board:
    def __init__(self):
        bw, gap = 250, 120
        x = (W - (3 * bw + 2 * gap)) // 2
        self.buttons = [Button(name, (x + i * (bw + gap), 28, x + i * (bw + gap) + bw, 128))
                        for i, name in enumerate(("ONE", "TWO", "THREE"))]
        self.cards = [Card("Card A", 170, 290, (60, 110, 225)), Card("Card B", W - 170 - Card.W, 290, (150, 160, 40))]
        self.slider = Slider(260, W - 260, 610)

    def target(self, p):
        """The object under point p, topmost first."""
        for card in reversed(self.cards):
            if card.hit(p):
                return card
        if self.slider.hit(p):
            return self.slider
        for button in self.buttons:
            if button.hit(p):
                return button
        return None

    def press(self, hand, now):
        obj = self.target(hand.pointer)
        if isinstance(obj, Button):
            obj.count += 1
            obj.flash = now
        elif isinstance(obj, (Card, Slider)) and obj.held_by is None:
            obj.held_by = hand
            hand.held = obj
            # Held things move with the palm, not the fingertips: the palm stays steady while
            # the fingers pinch and open, and stays sharp when the hand moves fast.
            if isinstance(obj, Card):
                self.cards.remove(obj)
                self.cards.append(obj)  # bring to front
                obj.vel[:] = 0          # catches a card in flight
                obj.offset = obj.pos - hand.palm.pos
            else:
                obj.offset = obj.knob_x - hand.palm.pos[0]

    def follow(self, hand):
        obj = hand.held
        if isinstance(obj, Card):
            x, y = hand.palm.pos + obj.offset
            obj.pos[:] = (np.clip(x, 0, W - Card.W), np.clip(y, 0, H - Card.H))
        elif isinstance(obj, Slider):
            x = hand.palm.pos[0] + obj.offset
            obj.value = float(np.clip((x - obj.x0) / (obj.x1 - obj.x0), 0, 1))

    def release(self, hand, throw=None):
        """Let go of what the hand holds. A card let go with a velocity (px/s) slides on."""
        obj = hand.held
        if obj is None:
            return
        obj.held_by = None
        hand.held = None
        if isinstance(obj, Card) and throw is not None:
            speed = float(np.linalg.norm(throw))
            if speed >= THROW_MIN:
                obj.vel[:] = throw * min(1.0, THROW_MAX / speed)

    def step(self, dt):
        """Slide thrown cards along: friction slows them, the window edges bounce them."""
        for c in self.cards:
            if c.held_by is not None or not c.vel.any():
                continue
            c.pos += c.vel * dt
            for i, hi in ((0, W - Card.W), (1, H - Card.H)):
                if c.pos[i] < 0 or c.pos[i] > hi:
                    c.pos[i] = np.clip(c.pos[i], 0, hi)
                    c.vel[i] *= -BOUNCE
            c.vel *= np.exp(-FRICTION * dt)
            if np.linalg.norm(c.vel) < 25:
                c.vel[:] = 0

    def draw(self, img, hands, now):
        hover = {}
        for hand in hands:
            if hand.visible and not hand.fist and hand.held is None:
                obj = self.target(hand.pointer)
                if obj is not None:
                    hover.setdefault(id(obj), hand.color)

        # Filled shapes go on a copy and are blended, so the camera shows through.
        overlay = img.copy()
        for b in self.buttons:
            fill = BUTTON_COLORS[b.count % len(BUTTON_COLORS)]
            if now - b.flash < 0.2:
                fill = lighten(fill, 0.6)
            rounded_rect(overlay, *b.rect, 20, fill)
        s = self.slider
        rounded_rect(overlay, s.x0 - 12, s.y - 12, s.x1 + 12, s.y + 12, 12, (60, 60, 60))
        rounded_rect(overlay, s.x0 - 12, s.y - 12, max(s.knob_x, s.x0 + 12) + 12, s.y + 12, 12, (80, 200, 120))
        for c in self.cards:
            x, y = c.pos
            if c.held_by is not None:
                rounded_rect(overlay, x + 10, y + 12, x + Card.W + 10, y + Card.H + 12, 16, (20, 20, 20))
            rounded_rect(overlay, x, y, x + Card.W, y + Card.H, 16, c.color)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, dst=img)

        # Outlines and text stay crisp on top.
        for b in self.buttons:
            x0, y0, x1, y1 = b.rect
            ring = hover.get(id(b))
            rounded_rect(img, x0, y0, x1, y1, 20, ring or WHITE, 5 if ring else 2)
            text(img, b.name, ((x0 + x1) / 2, y0 + 38), 1.0, align="center")
            text(img, f"pressed {b.count}", ((x0 + x1) / 2, y0 + 76), 0.7, align="center")

        ring = s.held_by.color if s.held_by else hover.get(id(s))
        kx = int(s.knob_x)
        cv2.circle(img, (kx, s.y), s.KNOB, WHITE, -1, AA)
        cv2.circle(img, (kx, s.y), s.KNOB, ring or (60, 60, 60), 5 if ring else 2, AA)
        text(img, f"{round(s.value * 100)}", (kx, s.y), 0.65, color=(40, 40, 40), align="center", outline=False)
        text(img, "SLIDER", (s.x0 - 12, s.y - 26), 0.6)
        text(img, f"value {round(s.value * 100)}", (s.x1 + 12, s.y - 26), 0.6, align="right")

        for c in self.cards:
            x, y = c.pos
            ring = c.held_by.color if c.held_by else hover.get(id(c))
            rounded_rect(img, x, y, x + Card.W, y + Card.H, 16, ring or WHITE, 5 if ring else 2)
            text(img, c.name, (x + Card.W / 2, y + 52), 1.0, align="center")
            if c.held_by is not None:
                text(img, f"held by {c.held_by.side.upper()}", (x + Card.W / 2, y + 100), 0.6,
                     color=lighten(c.held_by.color, 0.4), align="center")
            else:
                text(img, "pinch to drag", (x + Card.W / 2, y + 100), 0.6, align="center")


# --------------------------------------------------------------------------- hands

class OneEuro:
    """One Euro filter: heavy smoothing when slow (kills shake), light when fast (no lag)."""

    def __init__(self, min_cutoff=0.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(dt, self.min_cutoff + self.beta * np.linalg.norm(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def dist(a, b):
    return float(np.linalg.norm(a - b))


class Motion:
    """A smoothed point seen by the camera ~30 times a second, drawn at the window's frame
    rate slightly ahead along its motion, so it keeps up with the hand instead of trailing."""

    def __init__(self):
        self.filter = OneEuro(min_cutoff=POINTER_SMOOTHING, beta=POINTER_RESPONSE)
        self.target = np.zeros(2)   # smoothed position in the latest camera frame
        self.pos = np.zeros(2)      # drawn position
        self.vel = np.zeros(2)      # px/s
        self.t = None               # capture time of the latest camera frame
        self.drawn = None           # last draw time; None = jump straight to the target
        self.history = []           # recent (time, target), for the throw velocity

    def reset(self):
        self.filter.reset()
        self.t = self.drawn = None
        self.vel[:] = 0
        self.history.clear()

    def stop(self):
        """The hand is out of sight: stay where it was last seen, no looking ahead."""
        self.vel[:] = 0
        self.history.clear()

    def update(self, x, now):
        x = self.filter(x, now)
        if self.t is not None and now > self.t:
            self.vel = 0.5 * (x - self.target) / (now - self.t) + 0.5 * self.vel
        self.target, self.t = x, now
        self.history = [(t, p) for t, p in self.history if now - t < 0.2] + [(now, x)]

    def throw_velocity(self):
        """Average velocity over the last ~0.1 s (a single frame is too noisy to throw with)."""
        if len(self.history) < 2:
            return np.zeros(2)
        now, p = self.history[-1]
        t0, p0 = next(((t, q) for t, q in reversed(self.history) if now - t >= 0.1), self.history[0])
        return (p - p0) / max(now - t0, 1e-3)

    def advance(self, now):
        if self.t is None:
            return
        if self.drawn is None:
            self.pos, self.drawn = self.target.copy(), now
            return
        # Look ahead only when clearly moving, so a still hand does not shake.
        speed = float(np.linalg.norm(self.vel))
        ahead = self.vel * np.clip((speed - 150) / 450, 0, 1) * np.clip(now - self.t, 0, PREDICT)
        k = 1 - np.exp(-max(now - self.drawn, 0) / 0.012)
        self.drawn = now
        self.pos = self.pos + (self.target + ahead - self.pos) * k


class Hand:
    def __init__(self, side, color):
        self.side, self.color = side, color
        self.tip = Motion()     # the cursor: between the thumb tip and index tip
        self.palm = Motion()    # the middle of the palm: held things move with this
        self.visible = False
        self.last_seen = -1e9
        self.landmarks = None   # (21, 2) pixel coordinates
        self.pinch = False
        self.pinch_count = 0
        self.fist = False
        self.fist_count = 0
        self.held = None
        self.readout = ""       # live gesture numbers, shown with the d key

    @property
    def pointer(self):
        return self.tip.pos

    @property
    def status(self):
        if not self.visible:
            return "not visible"
        if self.fist:
            return "frozen"
        if self.held is not None:
            return f"dragging {self.held.name}"
        if self.pinch:
            return "pinching"
        return "moving"

    def update(self, det, now, board):
        """Advance this hand by one camera frame (det is None when the hand is not seen)."""
        if det is None:
            if self.visible:
                self.visible = False
                self.pinch_count = self.fist_count = 0
                self.tip.stop()
                self.palm.stop()
            if now - self.last_seen > LOST_GRACE:
                # Gone for real: drop whatever it held exactly where it is.
                board.release(self)
                self.pinch = self.fist = False
            return

        pts, world = det
        if not self.visible:
            self.tip.reset()
            self.palm.reset()
        self.visible, self.last_seen, self.landmarks = True, now, pts

        pinch = dist(pts[4], pts[8]) / (dist(pts[0], pts[9]) or 1e-6)
        reach = [dist(world[0], world[t]) / (dist(world[0], world[p]) or 1e-6) for t, p in FINGERS]
        self.readout = f"pinch {pinch:.2f}  index {reach[0]:.2f}"

        # A closed fist freezes this hand only: it drops what it holds and ignores the board
        # until it opens. The folded index finger is what tells a fist from a pinch made with
        # the other fingers tucked in; in a fist the thumb is not on the index tip.
        fist_shape = reach[0] < INDEX_CURLED and max(reach[1:]) < OTHERS_CURLED and pinch > FIST_THUMB_GAP
        if not self.fist:
            self.fist_count = self.fist_count + 1 if fist_shape and not self.pinch else 0
            if self.fist_count >= FIST_FRAMES:
                self.fist, self.fist_count = True, 0
                board.release(self)
        else:
            self.fist_count = self.fist_count + 1 if reach[0] > INDEX_OPEN else 0
            if self.fist_count >= DEBOUNCE:
                self.fist, self.fist_count = False, 0

        # The cursor is the pinch point itself, between the thumb tip and index tip, so you
        # grab things right where your fingers are. Smoothed so it does not shake.
        self.tip.update(np.clip((pts[4] + pts[8]) / 2, (0, 0), (W - 1, H - 1)), now)
        self.palm.update(pts[[0, 5, 9, 13, 17]].mean(0), now)
        if self.fist:
            return

        if not self.pinch:
            ready = pinch < PINCH_ON and reach[0] >= INDEX_CURLED  # a folding fist is not a pinch
            self.pinch_count = self.pinch_count + 1 if ready else 0
            if self.pinch_count >= DEBOUNCE:
                self.pinch, self.pinch_count = True, 0
                board.press(self, now)
        else:
            off = HOLD_OFF if self.held is not None else PINCH_OFF
            self.pinch_count = self.pinch_count + 1 if pinch > off else 0
            if self.pinch_count >= DEBOUNCE:
                self.pinch, self.pinch_count = False, 0
                board.release(self, throw=self.palm.throw_velocity())

    def advance(self, now):
        """Move the drawn cursor and palm on for this window frame."""
        self.tip.advance(now)
        self.palm.advance(now)

    def draw_skeleton(self, img, debug=False):
        if not self.visible:
            return
        pts = self.landmarks.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), self.color, 3, AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, WHITE, -1, AA)
        cv2.circle(img, tuple(pts[8]), 8, self.color, -1, AA)  # index fingertip
        cv2.circle(img, tuple(pts[8]), 8, WHITE, 2, AA)
        x0, y1, x1 = pts[:, 0].min(), pts[:, 1].max(), pts[:, 0].max()
        org = (np.clip((x0 + x1) / 2, 110, W - 110), min(y1 + 30, H - 50))
        text(img, f"{self.side.upper()}: {self.status}", org, 0.75, color=lighten(self.color, 0.3), align="center")
        if debug:
            text(img, self.readout, (org[0], org[1] + 28), 0.6, align="center")

    def draw_pointer(self, img):
        """The grab cursor, between thumb and index: a ring that fills in while pinching."""
        if not self.visible:
            return
        p = tuple(self.pointer.astype(int))
        cv2.circle(img, p, 14, BLACK, 6, AA)
        if self.pinch and not self.fist:
            cv2.circle(img, p, 14, self.color, -1, AA)
        cv2.circle(img, p, 14, self.color if not self.fist else GREY, 3, AA)
        if self.fist:  # "paused" bars
            cv2.line(img, (p[0] - 4, p[1] - 6), (p[0] - 4, p[1] + 6), WHITE, 3, AA)
            cv2.line(img, (p[0] + 4, p[1] - 6), (p[0] + 4, p[1] + 6), WHITE, 3, AA)


def read_hands(result, hands):
    """Map detections on the raw (unmirrored) frame to {"Left"/"Right": (pixel landmarks
    in the mirrored view, world landmarks)}.

    Detection runs on the raw frame because that is where the model's handedness labels
    are the person's real hands (checked against MediaPipe's own right-hand test images)."""
    dets = []
    for lms, world, handed in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
        pts = np.array([[(1 - p.x) * W, p.y * H] for p in lms])
        wpts = np.array([[p.x, p.y, p.z] for p in world])
        dets.append([handed[0].category_name, pts, wpts])
    if len(dets) == 2 and dets[0][0] == dets[1][0]:
        # Same label twice: in the mirrored view the user's left hand is on the left.
        dets.sort(key=lambda d: d[1][0, 0])
        dets[0][0], dets[1][0] = "Left", "Right"
    elif len(dets) == 1:
        # A lone hand's label can flicker; keep it with the hand that was just tracking there.
        d = dets[0]
        same, other = hands[d[0]], hands["Right" if d[0] == "Left" else "Left"]
        if other.visible and not same.visible and dist(other.landmarks[8], d[1][8]) < 0.15 * W:
            d[0] = other.side
    return {side: (pts, wpts) for side, pts, wpts in dets}


# --------------------------------------------------------------------------- app

def render(img, hands, board, now, fps, debug=False):
    board.draw(img, hands, now)
    for hand in hands:
        hand.draw_skeleton(img, debug)
    for hand in hands:
        hand.draw_pointer(img)
    left, right = hands
    text(img, f"LEFT: {left.status}", (16, H - 16), 0.7, color=lighten(left.color, 0.3))
    text(img, f"RIGHT: {right.status}", (W - 16, H - 16), 0.7, color=lighten(right.color, 0.3), align="right")
    text(img, f"pinch = press / grab    let go moving = throw    fist = freeze hand    f = fullscreen    d = numbers    q = quit    {fps:.0f} fps",
         (W / 2, H - 22), 0.5, color=(230, 230, 230), thick=1, align="center")


def fit(img):
    """Center-crop to 16:9 and scale to the window size."""
    h, w = img.shape[:2]
    if w / h > W / H:
        nw = int(h * W / H)
        img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    elif w / h < W / H:
        nh = int(w * H / W)
        img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    if img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"Downloading hand model to {MODEL_PATH.name} ...")
    tmp = MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(MODEL_PATH)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    deadline = time.monotonic() + 10  # leaves time to answer the macOS Camera prompt
    while time.monotonic() < deadline:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        else:
            cap.open(index, cv2.CAP_AVFOUNDATION)
        time.sleep(0.2)
    sys.exit(
        "Could not read from the camera.\n"
        "Allow camera access for your terminal app in System Settings > Privacy & Security > Camera,\n"
        "then quit and reopen the terminal and run this again."
    )


class Tracker(threading.Thread):
    """Reads the camera and runs hand detection on its own thread, so detection keeps up
    with every camera frame while the window redraws at its own, higher rate."""

    def __init__(self, cap, landmarker):
        super().__init__(daemon=True)
        self.cap, self.landmarker = cap, landmarker
        self.lock = threading.Lock()
        self.latest = None      # (frame number, capture time, raw frame, detection result)
        self.error = None
        self.running = True

    def run(self):
        count, last_ts, misses = 0, 0, 0
        while self.running:
            ok, raw = self.cap.read()
            if not ok or raw is None:
                misses += 1
                if misses > 60:
                    self.error = "The camera stopped delivering frames."
                    return
                time.sleep(0.01)
                continue
            misses = 0
            now = time.monotonic()
            raw = fit(raw)
            last_ts = max(int(now * 1000), last_ts + 1)
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
            count += 1
            with self.lock:
                self.latest = (count, now, raw, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    # Hand the GIL between the drawing and tracking threads every 1 ms instead of 5 ms;
    # otherwise drawing stalls detection long enough to miss camera frames.
    sys.setswitchinterval(0.001)
    ensure_model()
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    cap = open_camera(args.camera)
    tracker = Tracker(cap, landmarker)
    tracker.start()

    board = Board()
    left, right = Hand("Left", LEFT_COLOR), Hand("Right", RIGHT_COLOR)
    hands = (left, right)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, W, H)
    fullscreen = debug = False
    seen, raw, cam_dt, cam_prev, drawn = 0, None, 1 / 30, None, None

    try:
        while True:
            if tracker.error:
                sys.exit(tracker.error)
            with tracker.lock:
                latest = tracker.latest
            if latest is not None and latest[0] != seen:
                seen, captured, raw, result = latest
                if cam_prev is not None:
                    cam_dt = 0.9 * cam_dt + 0.1 * (captured - cam_prev)
                cam_prev = captured
                dets = read_hands(result, {"Left": left, "Right": right})
                for hand in hands:
                    hand.update(dets.get(hand.side), captured, board)

            if raw is not None:
                now = time.monotonic()
                board.step(min(now - drawn, 0.05) if drawn else 0.0)
                drawn = now
                for hand in hands:
                    hand.advance(now)
                    board.follow(hand)
                frame = cv2.flip(raw, 1)  # show it like a mirror
                render(frame, hands, board, now, 1 / cam_dt, debug)
                cv2.imshow(WINDOW, frame)

            # On macOS imshow + waitKey already wait for the display refresh (~60 Hz), so only
            # sleep here while there is nothing to draw yet.
            key = cv2.waitKey(1 if raw is not None else 10) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                debug = not debug
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if raw is not None and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.running = False
        tracker.join(timeout=1)
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
