"""""
Optimizations -mitodoescode

  Inference     Face/hand models run every N frames (results reused in between),
                Hand model skipped while no pose is detected; models run.
                simultaneously in worker threads; optional Lite pose model.

  Camera        explicit 640x480 @ 30, buffer size 1, grabbed on a background
                thread so the loop always works on the newest frame.

  Renderer      static ground grid precomputed (169 pts instead of 625,
                vectorised culling), canvas / occlusion buffers reused.

  Math          rigid inverse replaced by (p - t) @ R, duplicate conversions
                removed, per-frame landmark pixel coords computed once.

  Debug         third-person preview, skeleton overlay and face-mesh dots are
                individually switchable; per-model / per-stage timings (HUD).


Anti-jitter changes:

  * One Euro filters (heavy smoothing when still, low lag when moving) on the
    landmarks, finger offsets and the preview overlay, instead of a fixed EMA.

  * The *smoothed* landmarks now feed the head-pose fit (raw ones used to).

  * A stale Face Landmarker result is no longer replayed as a new measurement.

  * Learned face-vs-pose rotation bias: no jump when the face is covered/skipped.

  * Frame timestamps come from the camera thread, so filter dt is accurate.

  * Visibility hysteresis (landmarks don't flicker in/out), finger/pinch
    hysteresis (gesture text doesn't flip-flop).

Keys:  q quit | h hands on/off | f face on/off | s skeleton overlay
       m face-mesh dots | w third-person window | p profiling HUD
       r raw / smoothed preview overlay

CLI examples EXPERIMENTALS MWEHEHE(tested on arch linux with python 3.11.6 and opencv 4.8.0):
  python fpv_pipeline_faceLandmarker_stable.py --profile
  python fpv_pipeline_faceLandmarker_stable.py --smooth high     # calmer, more lag
  python fpv_pipeline_faceLandmarker_stable.py --smooth low      # snappier, more jitter
  python fpv_pipeline_faceLandmarker_stable.py --pose lite --face-every 1 --hand-every 2
"""

import argparse
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def pose_model(variant):
    name = f"pose_landmarker_{variant}.task"
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           f"pose_landmarker_{variant}/float16/1/{name}")
    return os.path.join(HERE, name), url


# ----------------------------- Landmark groups -----------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
HEAD_IDS = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
# Pose hand points: (wrist, pinky, index, thumb)
L_HAND = (15, 17, 19, 21)
R_HAND = (16, 18, 20, 22)
HAND_IDS = list(L_HAND + R_HAND)
USED_IDS = sorted(set(HEAD_IDS + HAND_IDS))
SIDES = (("L", L_HAND), ("R", R_HAND))   # L = the person's left hand

# Hand Landmarker topology (21 points)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm edge
]
FINGERTIPS = (4, 8, 12, 16, 20)
POSE_HAND_CONNECTIONS = [(0, 1), (0, 2), (0, 3)]  # fallback 4-point hand

HAND_COLORS = {"L": (80, 120, 255), "R": (255, 160, 80)}  # BGR

# Face Landmarker (478-point mesh) indices used to measure how much of the face
# is covered by hands. Eyes / nose bridge count double: they matter most.
FACE_EYE_IDX = {33, 133, 159, 145, 362, 263, 386, 374, 168, 1, 6}
FACE_KEY = [1, 6, 168, 197, 195, 5, 4, 0, 17, 13, 14, 61, 291,
            33, 133, 159, 145, 362, 263, 386, 374,
            70, 63, 105, 300, 293, 334, 10, 152, 234, 454, 127, 356]
FACE_KEY_W = np.array([2.0 if i in FACE_EYE_IDX else 1.0 for i in FACE_KEY])
FACE_KEY_W_SUM = float(FACE_KEY_W.sum())

# Generic head model in the *proper* head frame (meters, origin = eye midpoint):
#   x = person's LEFT, y = up, z = forward.  Adapted online to the real user.
HEAD_MODEL_DEFAULT = {
    NOSE:  (0.000, -0.045, 0.045),
    L_EYE: (0.032, 0.000, 0.000),
    R_EYE: (-0.032, 0.000, 0.000),
    L_EAR: (0.078, -0.020, -0.090),
    R_EAR: (-0.078, -0.020, -0.090),
}

# ------------------------------- Parameters --------------------------------
VIS_MIN = 0.5           # a landmark becomes active at this visibility ...
VIS_OFF = 0.35          # ... and only drops out below this (hysteresis: no flicker)
STALE_S = 0.5           # filters re-initialise after this long without data
HAND_HOLD_S = 0.25      # keep last finger data this long if the hand is lost
MATCH_MAX_DIST = 0.15   # max normalized image distance to pair hand<->wrist
MOVING_SPEED = 0.25     # m/s: wrist speed (relative to head) counted as moving
EYE_HEIGHT = 1.6
HFOV_DEG = 90.0
VIEW_W, VIEW_H = 640, 480

# ---- Performance profile (all overridable from the command line) ----
POSE_VARIANT = "full"     # "full" or "lite" (benchmark lite for speed)
FACE_EVERY_N = 1          # run Face Landmarker every N frames (1 = every frame; >1 = jitterier)
HAND_EVERY_N = 2          # run Hand Landmarker every N frames
PARALLEL_INFERENCE = True # run pose / hand / face concurrently
CAM_INDEX = 0
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
CAM_FOURCC = None         # e.g. "MJPG" can help some USB webcams
GROUND_RANGE = 12         # ground grid half-size (meters)
GROUND_STEP = 2           # grid spacing -> 13x13 = 169 points (was 625)

# ---- Head pose: confidence ----
VIS_LO, VIS_HI = 0.3, 0.8     # pose visibility ramp -> weight 0..1
W_MIN_USE = 0.05              # landmarks lighter than this are ignored
FACE_BASE_CONF = 0.95         # max confidence of a Face Landmarker measurement
FACE_OCC_POWER = 2.0          # face confidence = base * (1 - covered)^power
FACE_PRIOR = 2.0              # Face Landmarker is trusted this much more than pose
POSE_FIT_RMS = 0.03           # m: head-model fit residual that halves pose confidence
POS_SPREAD = 0.05             # m: landmark-vote disagreement that halves position confidence
FACE_MATRIX_TRANSPOSE = False  # flip to True if the console warns about face/pose disagreement
ADAPT_ALPHA = 0.02            # head model adaptation rate (clean frames only)

# ---- Head pose: gating against the prediction ----
GATE_BASE_DEG = 10.0          # innovation always accepted below this
GATE_COAST_DEG_S = 180.0      # ... plus this per second of coasting (uncertainty grows)
GATE_SOFT_DEG = 12.0          # how fast confidence falls beyond the gate
AGREE_DEG = 40.0              # face vs pose rotation disagreement -> keep only one
POS_GATE_BASE = 0.06          # m
POS_GATE_COAST = 0.5          # m per second of coasting
POS_GATE_SOFT = 0.06          # m

# ---- Head pose: adaptive smoothing / prediction ----
ROT_MIN_CUTOFF = 1.5          # Hz, smoothing when the head is still
ROT_BETA = 0.8                # Hz per rad/s: less smoothing when turning fast
POS_MIN_CUTOFF = 1.2          # Hz
POS_BETA = 4.0                # Hz per m/s
VEL_GAIN = 0.2                # velocity update rate (scaled by confidence)
OMEGA_TAU_S = 0.35            # angular velocity decay while coasting
VEL_TAU_S = 0.20              # linear velocity decay while coasting
MAX_TURN_DEG_S = 720.0        # clamp for angular velocity
MAX_CORR_DEG_S = 540.0        # max rotation correction per second (anti-snap)
MAX_HEAD_SPEED = 3.0          # m/s clamp for linear velocity

# ---- Face-vs-pose rotation bias (learned online) ----
BIAS_ALPHA = 0.05             # learning rate per frame (scaled by confidence)
BIAS_MIN_CONF = 0.5           # both sources must be at least this confident to learn
BIAS_LEARN_DEG = 25.0         # only learn while the two roughly agree

# ---- Jitter control: One Euro settings, chosen with --smooth ----
# (min_cutoff Hz while still, beta Hz per unit/s of speed). Lower min_cutoff =
# calmer when still; higher beta = less lag when moving.
SMOOTH_PRESETS = {
    "low": dict(LM_HEAD=(2.0, 10.0), LM_BODY=(3.0, 8.0), FINGER_F=(3.0, 8.0),
                PIX_F=(3.0, 6.0), ROT=(1.5, 0.8), POS=(1.2, 4.0)),
    "medium": dict(LM_HEAD=(1.2, 8.0), LM_BODY=(1.8, 6.0), FINGER_F=(2.0, 6.0),
                   PIX_F=(1.5, 6.0), ROT=(1.0, 0.8), POS=(0.8, 4.0)),
    "high": dict(LM_HEAD=(0.8, 8.0), LM_BODY=(1.2, 6.0), FINGER_F=(1.2, 5.0),
                 PIX_F=(1.0, 6.0), ROT=(0.7, 0.8), POS=(0.5, 4.0)),
}
LM_HEAD = LM_BODY = FINGER_F = PIX_F = (1.0, 5.0)    # set by apply_smoothing()


def apply_smoothing(name):
    global LM_HEAD, LM_BODY, FINGER_F, PIX_F
    global ROT_MIN_CUTOFF, ROT_BETA, POS_MIN_CUTOFF, POS_BETA
    p = SMOOTH_PRESETS[name]
    LM_HEAD, LM_BODY = p["LM_HEAD"], p["LM_BODY"]
    FINGER_F, PIX_F = p["FINGER_F"], p["PIX_F"]
    ROT_MIN_CUTOFF, ROT_BETA = p["ROT"]
    POS_MIN_CUTOFF, POS_BETA = p["POS"]


# ------------------------------ Core math ----------------------------------
def to_right_handed(lm):
    """MediaPipe world coords (x right, y DOWN, z away from camera, meters)
     right-handed frame (x right, y UP, z toward camera)."""
    return np.array([lm.x, -lm.y, -lm.z], dtype=float)


def invert_rigid(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def to_head_frame(points, T):
    """p' = T^-1 p = R^T (p - t)  (x right, y up, z forward, relative to eyes).
    Row-vector form (p - t) @ R: no homogeneous matrix, no 4x4 inverse."""
    return (np.asarray(points, dtype=float) - T[:3, 3]) @ T[:3, :3]


def project(points_head, width, height, hfov_deg=HFOV_DEG, near=0.05):
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    z = points_head[:, 2]
    safe_z = np.where(z > near, z, 1.0)
    sx = width / 2.0 + f * points_head[:, 0] / safe_z
    sy = height / 2.0 - f * points_head[:, 1] / safe_z
    visible = (z > near) & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
    return np.stack([sx, sy], axis=1), visible, f


# ------------------------------ Quaternions --------------------------------
# (w, x, y, z).  Used for slerp / prediction of the head rotation.
def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_matrix(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rotvec(q):
    if q[0] < 0:
        q = -q
    w = min(float(q[0]), 1.0)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        return 2.0 * q[1:]
    return q[1:] / s * (2.0 * np.arccos(w))


def rotvec_to_quat(v):
    ang = np.linalg.norm(v)
    if ang < 1e-9:
        return quat_normalize(np.array([1.0, v[0] / 2, v[1] / 2, v[2] / 2]))
    return np.concatenate([[np.cos(ang / 2)], v / ang * np.sin(ang / 2)])


def quat_angle(a, b):
    return 2.0 * float(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def quat_slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return quat_normalize(a + t * (b - a))
    th = np.arccos(d)
    return quat_normalize((np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th))


def soft_gate(excess, soft):
    """1.0 when excess <= 0, then falls off smoothly."""
    e = max(excess, 0.0) / soft
    return 1.0 / (1.0 + e * e)


def one_euro_alpha(cutoff_hz, dt):
    return 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)


MIRROR_X = np.diag([-1.0, 1.0, 1.0])


# ------------------------------ One Euro filter -----------------------------
class OneEuro:
    """Adaptive low-pass: cutoff = min_cutoff + beta * speed. Nearly frozen when
    the signal is still (remove s jitter), opens up when it moves (bit of lag).
    Works on arrays; `dim` is the size of one point (speed is the RMS point
    speed, so a 21-point hand uses one shared cutoff). Re-initialises after a
    gap longer than STALE_S so a re-acquired landmark snaps instead of gliding."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0, dim=3):
        self.min_cutoff, self.beta, self.d_cutoff, self.dim = min_cutoff, beta, d_cutoff, dim
        self.x = None
        self.dx = None
        self.t = None

    def __call__(self, t, x):
        x = np.asarray(x, dtype=float)
        if self.x is None or t - self.t > STALE_S or x.shape != self.x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x
        dt = max(t - self.t, 1e-3)
        a_d = one_euro_alpha(self.d_cutoff, dt)
        self.dx = a_d * ((x - self.x) / dt) + (1.0 - a_d) * self.dx
        speed = float(np.sqrt(np.mean(np.sum(self.dx.reshape(-1, self.dim) ** 2, axis=1))))
        a = one_euro_alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = a * x + (1.0 - a) * self.x          # new array: safe to hand out
        self.t = t
        return self.x


class PixelSmoother:
    """Smooths normalised image coordinates of the pose landmarks for the
    third-person overlay only ."""

    def __init__(self):
        self.f = {}

    def __call__(self, t, lms, ids, w, h):
        out = {}
        for i in ids:
            f = self.f.get(i)
            if f is None:
                f = self.f[i] = OneEuro(*PIX_F, dim=2)
            xy = f(t, (lms[i].x, lms[i].y))
            out[i] = (int(xy[0] * w), int(xy[1] * h))
        return out


# --------------------------- Landmark smoothing ----------------------------
class LandmarkTracker:
    """Per-landmark One Euro smoothing of the pose world coordinates, with
    visibility hysteresis. `raw` = latest detection, `pos` = smoothed/held."""

    def __init__(self):
        self.pos = {}
        self.vis = {}
        self.raw = {}
        self.active = {}
        self.filters = {}

    def update(self, world_landmarks, ids, t):
        for i in ids:
            lm = world_landmarks[i]
            v = lm.visibility if lm.visibility is not None else 1.0
            p = np.array([lm.x, -lm.y, -lm.z], dtype=float)   # converted once
            self.vis[i] = v
            self.raw[i] = p
            on = v >= (VIS_OFF if self.active.get(i, False) else VIS_MIN)
            self.active[i] = on
            if not on:
                continue                       # hold the last smoothed position
            f = self.filters.get(i)
            if f is None:
                cfg = LM_HEAD if i in HEAD_IDS else LM_BODY
                f = self.filters[i] = OneEuro(*cfg, dim=3)
            self.pos[i] = f(t, p)

    def visible(self, ids):
        return all(self.active.get(i, False) and i in self.pos for i in ids)


# --------------------------- Occlusion measurement -------------------------
@lru_cache(maxsize=16)
def _kernel(k):
    return np.ones((k, k), np.uint8)


class OcclusionMasker:
    """Soft low-res mask (0..1) of where hands are in the webcam image.
    Buffers and shape morphology kernels are reused between frames; returns None
    when there are no hands, skipping all the work."""

    def __init__(self, scale=4):
        self.scale = scale
        self._shape = None
        self._u8 = None
        self._tmp = None
        self._f32 = None

    def build(self, groups, w, h, grow_px):
        groups = [g for g in groups if g]
        if not groups:
            return None
        s = self.scale
        mw, mh = max(w // s, 1), max(h // s, 1)
        if self._shape != (mh, mw):
            self._shape = (mh, mw)
            self._u8 = np.zeros((mh, mw), np.uint8)
            self._tmp = np.zeros((mh, mw), np.uint8)
            self._f32 = np.zeros((mh, mw), np.float32)
        mask = self._u8
        mask.fill(0)
        for pts in groups:
            px = np.array(pts, dtype=np.float32) * np.array([mw, mh], np.float32)
            if len(px) >= 3:
                cv2.fillConvexPoly(mask, cv2.convexHull(px).astype(np.int32), 255)
            for x, y in px.astype(np.int32).tolist():
                cv2.circle(mask, (x, y), 1, 255, -1)
        k = int(grow_px / s) * 2 + 1
        if k > 1:
            cv2.dilate(mask, _kernel(k), dst=self._tmp)
            cv2.GaussianBlur(self._tmp, (0, 0), max(k / 4.0, 1.0), dst=mask)
        np.multiply(mask, 1.5 / 255.0, out=self._f32, dtype=np.float32)
        np.minimum(self._f32, 1.0, out=self._f32)
        return self._f32


def sample_mask(mask, xy):
    """Occlusion (0..1) at normalized image points xy (N,2). mask=None  zeros."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    if mask is None:
        return np.zeros(len(xy))
    mh, mw = mask.shape
    xs = np.clip((xy[:, 0] * mw).astype(int), 0, mw - 1)
    ys = np.clip((xy[:, 1] * mh).astype(int), 0, mh - 1)
    return mask[ys, xs]


def vis_weight(v):
    return float(np.clip((v - VIS_LO) / (VIS_HI - VIS_LO), 0.0, 1.0))


def face_rotation_from_matrix(M):
    """Proper 3x3 rotation (columns = face left, up, forward in camera axes:
    x right, y up, z toward camera) from the facial transformation matrix."""
    A = np.asarray(M, dtype=float)[:3, :3]
    if FACE_MATRIX_TRANSPOSE:
        A = A.T
    s = np.linalg.norm(A, axis=0)
    if np.any(s < 1e-6):
        return None
    U, _, Vt = np.linalg.svd(A / s)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# --------------------------- Head pose estimator ---------------------------
class HeadPoseEstimator:
    """Confidence-aware head pose: fusion + prediction + adaptive smoothing.

    Internal orientation is a *proper* rotation (columns: left, up, forward).
    `T` converts it to the convention the rest of the pipeline uses
    (columns: right, up, forward; translation = eye midpoint).
    """

    def __init__(self):
        self.model = {i: np.array(v, dtype=float) for i, v in HEAD_MODEL_DEFAULT.items()}
        self.q = None
        self.p = None
        self.omega = np.zeros(3)
        self.v = np.zeros(3)
        self.last_t = None
        self.coast_rot = 0.0
        self.coast_pos = 0.0
        self.info = {"conf": 0.0, "c_face": 0.0, "c_pose": 0.0, "c_pos": 0.0,
                     "mode": "starting"}
        self._agree = deque(maxlen=90)
        self._warned = False
        self.q_bias = np.array([1.0, 0.0, 0.0, 0.0])   # face ≈ q_bias * pose

    # ------------------------------ output ---------------------------------
    @property
    def T(self):
        if self.q is None or self.p is None:
            return None
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(self.q) @ MIRROR_X
        T[:3, 3] = self.p
        return T

    # --------------------------- measurements ------------------------------
    def _fit_pose(self, raw, w):
        """Weighted Kabsch fit of the head model to the trusted pose points."""
        ids = [i for i in HEAD_IDS if i in raw and w.get(i, 0.0) > W_MIN_USE]
        if len(ids) < 3:
            return None
        M = np.array([self.model[i] for i in ids])
        O = np.array([raw[i] for i in ids])
        ww = np.array([w[i] for i in ids])
        ws = ww.sum()
        mc = (ww[:, None] * M).sum(0) / ws
        oc = (ww[:, None] * O).sum(0) / ws
        H = ((M - mc) * ww[:, None]).T @ (O - oc)
        U, S, Vt = np.linalg.svd(H)
        if S[1] / ws < 2e-4:                      # near-collinear: ill-posed
            return None
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d if d != 0 else 1.0]) @ U.T
        t = oc - R @ mc
        res = O - (M @ R.T + t)
        rms = float(np.sqrt((ww * (res ** 2).sum(1)).sum() / ws))
        conf = float(np.clip((ws - 1.5) / 3.0, 0.0, 1.0) * np.exp(-(rms / POSE_FIT_RMS) ** 2))
        return quat_from_matrix(R), conf

    def _position_votes(self, raw, w, R):
        """Each trusted landmark votes for the eye-midpoint position."""
        ids = [i for i in HEAD_IDS if i in raw and w.get(i, 0.0) > W_MIN_USE]
        if not ids:
            return None
        votes = np.array([raw[i] - R @ self.model[i] for i in ids])
        ww = np.array([w[i] for i in ids])
        ws = ww.sum()
        mean = (ww[:, None] * votes).sum(0) / ws
        spread = float(np.sqrt((ww * ((votes - mean) ** 2).sum(1)).sum() / ws))
        conf = float(np.clip((ws - 0.5) / 2.0, 0.0, 1.0) * np.exp(-(spread / POS_SPREAD) ** 2))
        return mean, conf

    def _track_agreement(self, ang):
        self._agree.append(ang)
        if (not self._warned and len(self._agree) == self._agree.maxlen
                and np.median(self._agree) > np.radians(50)):
            self._warned = True
            print("WARNING: Face Landmarker and pose head rotations disagree a lot. "
                  "Try FACE_MATRIX_TRANSPOSE = True at the top of the script.")

    # ------------------------------- filters -------------------------------
    def _update_rot(self, dt, q_pred, q_meas, c_meas):
        if self.q is None:
            if q_meas is not None and c_meas > 0.3:
                self.q = q_meas.copy()
                self.omega = np.zeros(3)
                self.coast_rot = 0.0
            return
        if q_meas is not None and c_meas > 0.0:
            r = quat_to_rotvec(quat_mul(q_meas, quat_conj(q_pred)))   # innovation
            speed = float(np.linalg.norm(self.omega))
            alpha = one_euro_alpha(ROT_MIN_CUTOFF + ROT_BETA * speed, dt)
            corr = float(np.clip(alpha * c_meas, 0.0, 1.0)) * r
            lim = np.radians(MAX_CORR_DEG_S) * dt
            n = np.linalg.norm(corr)
            if n > lim:
                corr *= lim / n
            q_new = quat_normalize(quat_mul(rotvec_to_quat(corr), q_pred))
            om_meas = quat_to_rotvec(quat_mul(q_new, quat_conj(self.q))) / dt
            kv = VEL_GAIN * c_meas
            omega = (1 - kv) * self.omega + kv * om_meas
        else:
            q_new, omega = q_pred, self.omega
        omega = omega * np.exp(-dt / OMEGA_TAU_S * (1.0 - c_meas))    # coast gently
        mx = np.radians(MAX_TURN_DEG_S)
        n = np.linalg.norm(omega)
        if n > mx:
            omega *= mx / n
        self.q, self.omega = q_new, omega
        self.coast_rot = (self.coast_rot + dt) * (1.0 - min(c_meas, 1.0))

    def _update_pos(self, dt, p_meas, c_pos):
        if self.p is None:
            if p_meas is not None and c_pos > 0.3:
                self.p = p_meas.copy()
                self.v = np.zeros(3)
                self.coast_pos = 0.0
            return
        p_pred = self.p + self.v * dt
        if p_meas is not None and c_pos > 0.0:
            allowed = POS_GATE_BASE + POS_GATE_COAST * self.coast_pos + 0.5 * MAX_HEAD_SPEED * dt
            c_pos *= soft_gate(float(np.linalg.norm(p_meas - p_pred)) - allowed, POS_GATE_SOFT)
            speed = float(np.linalg.norm(self.v))
            alpha = one_euro_alpha(POS_MIN_CUTOFF + POS_BETA * speed, dt)
            p_new = p_pred + float(np.clip(alpha * c_pos, 0.0, 1.0)) * (p_meas - p_pred)
            kv = VEL_GAIN * c_pos
            v = (1 - kv) * self.v + kv * (p_new - self.p) / dt
        else:
            c_pos = 0.0
            p_new, v = p_pred, self.v
        v = v * np.exp(-dt / VEL_TAU_S * (1.0 - c_pos))
        n = np.linalg.norm(v)
        if n > MAX_HEAD_SPEED:
            v *= MAX_HEAD_SPEED / n
        self.p, self.v = p_new, v
        self.coast_pos = (self.coast_pos + dt) * (1.0 - min(c_pos, 1.0))
        self.info["c_pos"] = c_pos

    def _adapt_model(self, raw, w):
        if self.q is None or self.p is None:
            return
        if any(w.get(i, 0.0) < 0.8 or i not in raw for i in HEAD_IDS):
            return
        R = quat_to_matrix(self.q)
        for i in HEAD_IDS:
            obs = R.T @ (raw[i] - self.p)
            self.model[i] = (1 - ADAPT_ALPHA) * self.model[i] + ADAPT_ALPHA * obs
        mid = (self.model[L_EYE] + self.model[R_EYE]) / 2.0     # keep origin at the eyes
        for i in HEAD_IDS:
            self.model[i] = self.model[i] - mid

    # -------------------------------- update -------------------------------
    def update(self, t, raw, weights, face_R=None, face_conf=0.0):
        """raw: {id: xyz world}, weights: {id: 0..1}, face_R: proper 3x3 or None,
        face_conf: 0..1.  Returns T (4x4) or None until the head is initialized."""
        dt = 1.0 / 30.0 if self.last_t is None else float(np.clip(t - self.last_t, 1e-3, 0.25))
        self.last_t = t

        q_pred = None
        if self.q is not None:
            q_pred = quat_normalize(quat_mul(rotvec_to_quat(self.omega * dt), self.q))

        q_face = quat_from_matrix(face_R) if (face_R is not None and face_conf > 0) else None
        c_face = float(face_conf) if q_face is not None else 0.0
        fit = self._fit_pose(raw, weights)
        q_pose, c_pose = (fit if fit is not None else (None, 0.0))

        # Face and pose rarely agree perfectly: there is a small, nearly constant
        # rotation offset between them. Learn it while both are trustworthy and
        # apply it to pose-only frames, otherwise the output jumps by that offset
        # every time the face is covered by a hand or skipped (visible twitching).
        if q_pose is not None:
            if (q_face is not None and c_face > BIAS_MIN_CONF and c_pose > BIAS_MIN_CONF
                    and quat_angle(q_face, quat_mul(self.q_bias, q_pose))
                    < np.radians(BIAS_LEARN_DEG)):
                target = quat_mul(q_face, quat_conj(q_pose))
                self.q_bias = quat_slerp(self.q_bias, target,
                                         BIAS_ALPHA * min(c_face, c_pose))
            q_pose = quat_mul(self.q_bias, q_pose)

        # soft-gate each measurement against the prediction
        if q_pred is not None:
            allowed = (np.radians(GATE_BASE_DEG) + np.radians(GATE_COAST_DEG_S) * self.coast_rot
                       + np.radians(0.25 * MAX_TURN_DEG_S) * dt)
            soft = np.radians(GATE_SOFT_DEG)
            if q_face is not None:
                c_face *= soft_gate(quat_angle(q_face, q_pred) - allowed, soft)
            if q_pose is not None:
                c_pose *= soft_gate(quat_angle(q_pose, q_pred) - allowed, soft)

        # the two sources contradict each other -> keep the more trusted one
        if q_face is not None and q_pose is not None and c_face > 0.05 and c_pose > 0.05:
            ang = quat_angle(q_face, q_pose)
            self._track_agreement(ang)
            if ang > np.radians(AGREE_DEG):
                if FACE_PRIOR * c_face >= c_pose:
                    c_pose, c_face = 0.0, c_face * 0.7
                else:
                    c_face, c_pose = 0.0, c_pose * 0.7

        wf = FACE_PRIOR * c_face if q_face is not None else 0.0
        wp = c_pose if q_pose is not None else 0.0
        if wf + wp > 1e-6:
            if wp <= 1e-9:
                q_meas = q_face
            elif wf <= 1e-9:
                q_meas = q_pose
            else:
                q_meas = quat_slerp(q_pose, q_face, wf / (wf + wp))
            c_meas = 1.0 - (1.0 - min(c_face, 1.0)) * (1.0 - min(c_pose, 1.0))
        else:
            q_meas, c_meas = None, 0.0

        self.info.update(c_face=c_face, c_pose=c_pose, conf=c_meas, c_pos=0.0)
        self._update_rot(dt, q_pred, q_meas, c_meas)

        # position: landmark votes through the (filtered) orientation
        p_meas, c_pos = None, 0.0
        if self.q is not None:
            votes = self._position_votes(raw, weights, quat_to_matrix(self.q))
            if votes is not None:
                p_meas, c_pos = votes
        self._update_pos(dt, p_meas, c_pos)

        if c_meas > 0.8 and (q_face is None or q_pose is None
                             or quat_angle(q_face, q_pose) < np.radians(20)):
            self._adapt_model(raw, weights)

        c = self.info["conf"]
        if self.q is None or self.p is None:
            self.info["mode"] = "starting"
        elif c >= 0.6:
            self.info["mode"] = "tracking"
        elif c >= 0.25:
            self.info["mode"] = "low confidence (blending prediction)"
        else:
            self.info["mode"] = "predicting"
        return self.T


# ------------------------------ Hands / fingers ----------------------------
def match_hands(hand_img_lms, pose_img_lms, wrist_ok):
    """Pair detected hands with pose wrists by image-space proximity.

    We do NOT trust the Left/Right label (it assumes a mirrored image).
    Returns {"L": hand_index, "R": hand_index}.
    """
    cands = []
    for hi, hl in enumerate(hand_img_lms):
        for side, group in SIDES:
            if not wrist_ok[side]:
                continue
            w = pose_img_lms[group[0]]
            d = np.hypot(hl[0].x - w.x, hl[0].y - w.y)
            if d < MATCH_MAX_DIST:
                cands.append((d, hi, side))
    cands.sort()
    used_h, used_s, out = set(), set(), {}
    for d, hi, side in cands:
        if hi in used_h or side in used_s:
            continue
        out[side] = hi
        used_h.add(hi)
        used_s.add(side)
    return out


def analyze_hand(off, prev=None):
    """off: (21,3) hand points. Returns (gesture, finger_states, pinch_m).
    `prev` (the previous result) adds hysteresis: a finger flips to extended
    only above the enter ratio and back to curled only below the lower exit
    ratio, so values sitting on a threshold stop flip-flopping."""
    def d(a, b):
        return float(np.linalg.norm(off[a] - off[b]))

    prev_g = prev[0] if prev else None
    prev_f = prev[1] if prev else {}

    def extended(name, tip, joint, ref, enter, leave):
        ratio = d(tip, ref) / max(d(joint, ref), 1e-9)
        return ratio > (leave if prev_f.get(name) else enter)

    palm = max(d(0, 9), 1e-6)
    fingers = {
        "thumb": extended("thumb", 4, 3, 17, 1.14, 1.06),
        "index": extended("index", 8, 6, 0, 1.09, 1.01),
        "middle": extended("middle", 12, 10, 0, 1.09, 1.01),
        "ring": extended("ring", 16, 14, 0, 1.09, 1.01),
        "pinky": extended("pinky", 20, 18, 0, 1.09, 1.01),
    }
    pinch = d(4, 8)
    ext = [fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]

    if pinch < (0.45 if prev_g == "PINCH" else 0.35) * palm:
        gesture = "PINCH"
    elif ext == [True, True, True, True]:
        gesture = "OPEN"
    elif ext == [True, False, False, False]:
        gesture = "POINT"
    elif ext == [True, True, False, False]:
        gesture = "PEACE"
    elif not any(ext):
        gesture = "THUMB" if fingers["thumb"] else "FIST"
    else:
        gesture = f"{sum(fingers.values())} FINGERS"
    return gesture, fingers, pinch


class HandTracker:
    """Smooths finger offsets and tracks wrist speed relative to the users  head."""

    def __init__(self):
        self.offsets = {}     # side -> (21,3) smoothed, wrist at origin
        self.last_seen = {}
        self.prev = {}        # side -> (t, wrist_in_head)
        self.speed = {"L": 0.0, "R": 0.0}
        self.info = {}        # side -> (gesture, fingers, pinch)
        self.filt = {}        # side -> One Euro filter on the finger offsets

    def update_fingers(self, side, hand_world, t):
        off = np.array([[l.x, -l.y, -l.z] for l in hand_world], dtype=float)
        off = off - off[0]
        f = self.filt.get(side)
        if f is None:
            f = self.filt[side] = OneEuro(*FINGER_F, dim=3)
        self.offsets[side] = f(t, off)
        self.last_seen[side] = t
        self.info[side] = analyze_hand(self.offsets[side], self.info.get(side))

    def touch(self, side, t):
        """Detection was reused this frame (hand model skipped): keep it fresh
        without re-running smoothing / gesture analysis on identical data."""
        if side in self.offsets:
            self.last_seen[side] = t

    def fresh(self, side, t):
        return side in self.offsets and t - self.last_seen[side] < HAND_HOLD_S

    def update_speed(self, side, wrist_head, t):
        if side in self.prev:
            t0, w0 = self.prev[side]
            dt = t - t0
            if dt > 1e-3:
                v = np.linalg.norm(wrist_head - w0) / dt
                self.speed[side] = 0.7 * self.speed[side] + 0.3 * v
        self.prev[side] = (t, wrist_head)


# -------------------------- First-person renderer --------------------------
class FirstPersonView:
    def __init__(self):
        self.cam = np.array([0.0, 0.0, 0.0])   # fixed: the camera no longer travels
        # The camera never moves, so the ground grid is built exactly once.
        rng = np.arange(-GROUND_RANGE, GROUND_RANGE + 1, GROUND_STEP, dtype=float)
        GX, GZ = np.meshgrid(rng + np.floor(self.cam[0]), rng + np.floor(self.cam[2]))
        ground = np.stack(
            [GX.ravel(), np.full(GX.size, -EYE_HEIGHT), GZ.ravel()], axis=1)
        self.ground = ground - self.cam
        self.bg = np.empty((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        self.bg[:] = (60, 40, 30)
        self.canvas = np.empty_like(self.bg)

    def render(self, T, hands):
        """hands: {side: (points_world (N,3), connections, kind)}"""
        np.copyto(self.canvas, self.bg)
        img = self.canvas
        R = T[:3, :3]

        # Ground dots: project, cull in numpy, loop only over what is visible
        view = self.ground @ R
        px, vis, _ = project(view, VIEW_W, VIEW_H)
        idx = np.nonzero(vis)[0]
        if idx.size:
            xs = px[idx, 0].astype(int).tolist()
            ys = px[idx, 1].astype(int).tolist()
            rs = np.clip(6.0 / np.maximum(view[idx, 2], 0.5), 1, 4).astype(int).tolist()
            for x, y, r in zip(xs, ys, rs):
                cv2.circle(img, (x, y), r, (110, 160, 110), -1)

        # Hands and fingers (real tracked positions, head-relative)
        for side, (pts, conns, kind) in hands.items():
            color = HAND_COLORS[side]
            ph = to_head_frame(pts, T)
            px, vis, f = project(ph, VIEW_W, VIEW_H)
            pxi = px.astype(int).tolist()
            vis_l = vis.tolist()
            for a, b in conns:
                if vis_l[a] and vis_l[b]:
                    cv2.line(img, tuple(pxi[a]), tuple(pxi[b]), color, 3)
            n = len(ph)
            is_full = kind == "full"
            base = 0.010 if is_full else 0.03
            mult = np.ones(n)
            if is_full:
                mult[list(FINGERTIPS)] = 1.4
            rad = np.clip(base * f / np.maximum(ph[:, 2], 1e-6) * mult, 2, 25).astype(int).tolist()
            for j in range(n):
                if not vis_l[j]:
                    continue
                tip = is_full and j in FINGERTIPS
                cv2.circle(img, tuple(pxi[j]), rad[j],
                           (255, 255, 255) if tip else color, -1)

        cv2.drawMarker(img, (VIEW_W // 2, VIEW_H // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 12, 1)
        return img


# ----------------------- Camera / inference / profiling --------------------
class ThreadedCamera:
    """Grabs frames on a background thread; read() hands over the newest one.
    Processing never works on stale buffered frames and capture overlaps with
    inference instead of blocking the loop."""

    def __init__(self, index, width, height, fps, fourcc=None):
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            self.cap = cv2.VideoCapture(index)
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cond = threading.Condition()
        self._frame, self._id, self._t = None, 0, 0.0
        self._alive, self._stop = True, False
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)

    @property
    def opened(self):
        return self.cap.isOpened()

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            t = time.monotonic()               # capture time, not processing time
            with self._cond:
                if ok:
                    self._frame, self._id, self._t = frame, self._id + 1, t
                else:
                    self._alive = False
                self._cond.notify_all()
            if not ok:
                break

    def read(self, last_id, timeout=2.0):
        """Block until a frame newer than last_id exists (ok, frame, id, t_capture)."""
        with self._cond:
            self._cond.wait_for(lambda: self._id != last_id or not self._alive, timeout)
            if self._id == last_id:
                return False, None, last_id, 0.0
            return True, self._frame, self._id, self._t

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


class InferenceRunner:
    """Runs the MediaPipe models of one frame concurrently. The first job runs
    on the calling thread, the rest on workers (MediaPipe releases the GIL
    during inference). Every landmarker is only ever used by one thread at a
    time because we wait for all jobs before the next frame."""

    def __init__(self, parallel):
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mp") if parallel else None

    def run(self, jobs):
        items = list(jobs.items())
        if self.pool is None or len(items) < 2:
            return {k: fn() for k, fn in items}
        futs = {k: self.pool.submit(fn) for k, fn in items[1:]}
        out = {}
        try:
            out[items[0][0]] = items[0][1]()
        finally:
            for k, fu in futs.items():
                out[k] = fu.result()
        return out

    def shutdown(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


class Profiler:
    """Cheap EMA timers: per-model inference time plus per-stage loop time."""

    def __init__(self):
        self.avg = {}
        self._t = time.perf_counter()

    def _ema(self, name, ms):
        self.avg[name] = 0.9 * self.avg.get(name, ms) + 0.1 * ms

    def start(self):
        self._t = time.perf_counter()

    def mark(self, name):
        n = time.perf_counter()
        self._ema(name, (n - self._t) * 1000.0)
        self._t = n

    def timed(self, name, fn):
        def run():
            t0 = time.perf_counter()
            r = fn()
            self._ema(name, (time.perf_counter() - t0) * 1000.0)
            return r
        return run

    def lines(self):
        a = self.avg
        models = " ".join(f"{k}:{a[k]:.1f}" for k in ("pose", "hand", "face") if k in a)
        stages = " ".join(f"{k}:{a[k]:.1f}" for k in
                          ("wait", "infer", "fuse", "draw3p", "render", "show") if k in a)
        return f"models ms  {models}", f"stages ms  {stages}"


# ---------------------------------- Setup ----------------------------------
def ensure_model(path, url, label):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    print(f"Downloading {label} ...")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)          # atomic: never leaves a truncated model
        print("Done.")
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"Could not download the {label} ({e}).")
        print(f"Download it manually from:\n  {url}\nand save it as:\n  {path}")
        return False


def conf_color(c):
    """BGR: red (0) -> green (1)."""
    c = float(np.clip(c, 0.0, 1.0))
    return (0, int(255 * c), int(255 * (1.0 - c)))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pose", choices=("full", "lite"), default=POSE_VARIANT,
                    help="pose model variant (lite is faster)")
    ap.add_argument("--face-every", type=int, default=FACE_EVERY_N,
                    help="run Face Landmarker every N frames")
    ap.add_argument("--hand-every", type=int, default=HAND_EVERY_N,
                    help="run Hand Landmarker every N frames")
    ap.add_argument("--res", default=f"{CAM_W}x{CAM_H}", help="camera resolution WxH")
    ap.add_argument("--camera", type=int, default=CAM_INDEX)
    ap.add_argument("--no-parallel", action="store_true", help="run models sequentially")
    ap.add_argument("--no-preview", action="store_true", help="hide the third-person window")
    ap.add_argument("--profile", action="store_true", help="show timing HUD")
    ap.add_argument("--smooth", choices=tuple(SMOOTH_PRESETS), default="medium",
                    help="jitter filtering strength (high = calmer but more lag)")
    ap.add_argument("--raw-overlay", action="store_true",
                    help="draw the raw (unsmoothed) detections on the preview")
    return ap.parse_args()


# ---------------------------------- Main -----------------------------------
def main():
    args = parse_args()
    apply_smoothing(args.smooth)
    import mediapipe as mp

    face_n = max(1, args.face_every)
    hand_n = max(1, args.hand_every)
    hand_phase = 1 % hand_n          # offset hand/face so the load alternates
    cam_w, cam_h = (int(v) for v in args.res.lower().split("x"))

    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions

    pose_path, pose_url = pose_model(args.pose)
    if not ensure_model(pose_path, pose_url, os.path.basename(pose_path)):
        return  # the pose model is required

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=pose_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    hand_landmarker = None
    if ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand_landmarker.task"):
        hand_opts = vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
    else:
        print("Continuing with the 4-point pose hands only.")
    fingers_on = hand_landmarker is not None

    face_landmarker = None
    if ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL, "face_landmarker.task"):
        face_opts = vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            # permissive thresholds: we weigh confidence ourselves instead of
            # letting MediaPipe silently drop the face when it is half covered
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
    else:
        print("Continuing with pose-only head tracking.")
    face_on = face_landmarker is not None

    cam = ThreadedCamera(args.camera, cam_w, cam_h, CAM_FPS, CAM_FOURCC)
    if not cam.opened:
        print("Error: Could not open webcam.")
        for lm in (hand_landmarker, face_landmarker):
            if lm is not None:
                lm.close()
        raise SystemExit
    cam.start()

    tracker = LandmarkTracker()
    hands_tr = HandTracker()
    fpv = FirstPersonView()
    head_est = HeadPoseEstimator()
    masker = OcclusionMasker()
    pix_smooth = PixelSmoother()
    hand2d_f = {s: OneEuro(*PIX_F, dim=2) for s, _ in SIDES}   # preview hand skeletons
    hand2d = {}
    prof = Profiler()
    infer = InferenceRunner(PARALLEL_INFERENCE and not args.no_parallel)
    waiting_img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    cv2.putText(waiting_img, "Waiting for head landmarks...", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # visual switches (tracking is unaffected by any of them)
    show_third = not args.no_preview
    show_skel = True
    show_mesh = True
    show_prof = args.profile
    smooth_overlay = not args.raw_overlay

    last_T = None
    start = time.monotonic()
    last_ts_ms = -1
    prev_t = start
    fps = 0.0
    frame_idx = 0
    cam_id = 0
    last_pose_ok = True
    hand_cache = None        # last Hand Landmarker result (reused between runs)
    face_cache = None        # last (face_pts, face_R)
    last_print = start

    third_lines = [(0, 2), (2, 7), (0, 5), (5, 8),
                   (15, 17), (15, 19), (15, 21),
                   (16, 18), (16, 20), (16, 22)]

    try:
        with vision.PoseLandmarker.create_from_options(pose_opts) as landmarker:
            while True:
                prof.start()
                ok, frame, cam_id, t_cap = cam.read(cam_id)
                if not ok:
                    print("Error: Could not read webcam frame.")
                    break
                prof.mark("wait")

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # camera capture time: steady spacing for every filter, no matter
                # how long inference took (it differs on face / hand frames)
                now = t_cap
                ts_ms = max(int((now - start) * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # ---------------- inference (scheduled, concurrent) ----------------
                run_face = face_on and frame_idx % face_n == 0
                # hand model is pointless without a pose to pair it with
                run_hand = fingers_on and last_pose_ok and frame_idx % hand_n == hand_phase
                frame_idx += 1
                jobs = {"pose": prof.timed(
                    "pose", lambda: landmarker.detect_for_video(image, ts_ms))}
                if run_hand:
                    jobs["hand"] = prof.timed(
                        "hand", lambda: hand_landmarker.detect_for_video(image, ts_ms))
                if run_face:
                    jobs["face"] = prof.timed(
                        "face", lambda: face_landmarker.detect_for_video(image, ts_ms))
                res = infer.run(jobs)
                prof.mark("infer")

                result = res["pose"]
                if run_hand:
                    hand_cache = res["hand"]
                if run_face:
                    fres = res["face"]
                    face_cache = None
                    if fres is not None and fres.face_landmarks:
                        fl = fres.face_landmarks[0]
                        pts = np.array([[fl[i].x, fl[i].y] for i in FACE_KEY])
                        mats = fres.facial_transformation_matrixes
                        fR = (face_rotation_from_matrix(mats[0])
                              if mats is not None and len(mats) > 0 else None)
                        face_cache = (pts, fR)
                if not face_on:
                    face_cache = None
                if not fingers_on:
                    hand_cache = None
                hand_res = hand_cache

                hands_out = {}
                status = "No pose detected"

                pose_ok = bool(result.pose_world_landmarks and result.pose_landmarks)
                last_pose_ok = pose_ok
                if not pose_ok:
                    hand_cache = hand_res = None
                img_lm = None
                if pose_ok:
                    world = result.pose_world_landmarks[0]
                    img_lm = result.pose_landmarks[0]
                    tracker.update(world, USED_IDS, now)

                # ================= HEAD: measurements + confidence =================
                face_pts, face_R = face_cache if face_cache is not None else (None, None)

                # head size in pixels (to scale the occlusion margin)
                if face_pts is not None:
                    ext = face_pts.max(0) - face_pts.min(0)
                    head_px = max(ext[0] * w, ext[1] * h)
                elif pose_ok:
                    hx = [img_lm[i].x for i in HEAD_IDS]
                    hy = [img_lm[i].y for i in HEAD_IDS]
                    head_px = max((max(hx) - min(hx)) * w, (max(hy) - min(hy)) * h)
                else:
                    head_px = 0.15 * w
                head_px = max(head_px, 0.05 * w)

                # where are the hands in the image?
                hand_groups = []
                if pose_ok:
                    for side, group in SIDES:
                        if tracker.vis.get(group[0], 0.0) >= VIS_MIN:
                            hand_groups.append([(img_lm[i].x, img_lm[i].y) for i in group])
                if hand_res is not None:
                    for hl in hand_res.hand_landmarks:
                        hand_groups.append([(q.x, q.y) for q in hl])
                occ_mask = masker.build(hand_groups, w, h, grow_px=0.25 * head_px)

                # face confidence from how much of the face is covered
                c_face, face_occ_pts = 0.0, None
                if face_pts is not None and face_R is not None:
                    face_occ_pts = sample_mask(occ_mask, face_pts)
                    face_occ = float((face_occ_pts * FACE_KEY_W).sum() / FACE_KEY_W_SUM)
                    c_face = FACE_BASE_CONF * (1.0 - face_occ) ** FACE_OCC_POWER

                # per-landmark weights for the pose head points
                head_raw, head_w = {}, {}
                if pose_ok:
                    occs = sample_mask(occ_mask, [[img_lm[i].x, img_lm[i].y] for i in HEAD_IDS])
                    for k, i in enumerate(HEAD_IDS):
                        # smoothed landmark (raw only while it is not yet trusted)
                        head_raw[i] = tracker.pos[i] if tracker.active.get(i) else tracker.raw[i]
                        head_w[i] = vis_weight(tracker.vis[i]) * (1.0 - float(occs[k])) ** 2

                # a replayed (stale) face rotation would tug the filter backwards,
                # so only a fresh Face Landmarker result counts as a measurement
                fresh_face = run_face and face_R is not None
                last_T = head_est.update(now, head_raw, head_w,
                                         face_R if fresh_face else None,
                                         c_face if fresh_face else 0.0)
                hinfo = head_est.info
                head_col = conf_color(hinfo["conf"])

                # pixel coordinates of the pose landmarks, computed once
                pxy = None
                if pose_ok:
                    if smooth_overlay:
                        pxy = pix_smooth(now, img_lm, USED_IDS, w, h)
                    else:
                        pxy = {i: (int(img_lm[i].x * w), int(img_lm[i].y * h))
                               for i in USED_IDS}
                    status = f"Head {hinfo['mode']}"

                    # ---- Fingers: pair detections with pose wrists ----
                    if hand_res is not None and hand_res.hand_landmarks:
                        wrist_ok = {s: tracker.visible([g[0]]) for s, g in SIDES}
                        pairs = match_hands(hand_res.hand_landmarks, img_lm, wrist_ok)
                        for side, hi in pairs.items():
                            if run_hand:
                                hands_tr.update_fingers(
                                    side, hand_res.hand_world_landmarks[hi], now)
                                arr = np.array([[p.x, p.y] for p in hand_res.hand_landmarks[hi]])
                                hand2d[side] = hand2d_f[side](now, arr) if smooth_overlay else arr
                            else:
                                hands_tr.touch(side, now)
                            if show_third and show_skel and side in hand2d:
                                hp = (hand2d[side] * (w, h)).astype(int).tolist()
                                col = HAND_COLORS[side]
                                for a, b in HAND_CONNECTIONS:
                                    cv2.line(frame, hp[a], hp[b], col, 2)

                    # ---- Build hand geometry for the first-person view ----
                    for side, group in SIDES:
                        wid = group[0]
                        if not tracker.visible([wid]):
                            continue
                        if fingers_on and hands_tr.fresh(side, now):
                            pts = tracker.pos[wid] + hands_tr.offsets[side]
                            hands_out[side] = (pts, HAND_CONNECTIONS, "full")
                        elif tracker.visible(list(group)):
                            pts = np.array([tracker.pos[i] for i in group])
                            hands_out[side] = (pts, POSE_HAND_CONNECTIONS, "pose")

                        # hand movement relative to the head
                        if last_T is not None:
                            wh = to_head_frame(tracker.pos[wid][None], last_T)[0]
                            hands_tr.update_speed(side, wh, now)
                prof.mark("fuse")

                # ---- Third-person overlay (colour = confidence, red -> green) ----
                if show_third:
                    if pose_ok and show_skel:
                        for a, b in third_lines:
                            cv2.line(frame, pxy[a], pxy[b],
                                     head_col if a in HEAD_IDS else (0, 255, 0), 2)
                        for i in USED_IDS:
                            if i in HEAD_IDS:
                                color = conf_color(head_w.get(i, 0.0))
                            elif i in HAND_IDS:
                                color = (0, 165, 255)
                            else:
                                color = (255, 200, 0)
                            cv2.circle(frame, pxy[i], 5, color, -1)

                    # face mesh key points: cyan = clear, red = covered by a hand
                    if face_pts is not None and show_mesh:
                        fxy = (face_pts * (w, h)).astype(int).tolist()
                        for k, (fx, fy) in enumerate(fxy):
                            occd = face_occ_pts is not None and face_occ_pts[k] > 0.5
                            cv2.circle(frame, (fx, fy), 2,
                                       (0, 0, 255) if occd else (255, 255, 0), -1)

                    cv2.putText(frame,
                                f"{status} | FPS {fps:.0f} | fingers {'ON' if fingers_on else 'OFF'}"
                                f" | face {'ON' if face_on else 'OFF'}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                prof.mark("draw3p")

                # ---- First-person view ----
                if last_T is not None:
                    fp_img = fpv.render(last_T, hands_out)
                else:
                    fp_img = waiting_img.copy()

                # ---- HUD ----
                cv2.putText(fp_img,
                            f"HEAD: {hinfo['mode']}  conf={hinfo['conf']:.2f} "
                            f"face={hinfo['c_face']:.2f} pose={hinfo['c_pose']:.2f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_col, 2)
                if show_prof:
                    l1, l2 = prof.lines()
                    cv2.putText(fp_img, f"FPS {fps:.0f}  {l1}", (10, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    cv2.putText(fp_img, l2, (10, 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    if now - last_print > 5.0:
                        last_print = now
                        print(f"FPS {fps:.0f} | {l1} | {l2}")
                y = VIEW_H - 15
                for side, _ in SIDES:
                    if side not in hands_out:
                        continue
                    spd = hands_tr.speed[side]
                    move = "moving" if spd > MOVING_SPEED else "still"
                    if hands_out[side][2] == "full" and side in hands_tr.info:
                        g, fing, pinch = hands_tr.info[side]
                        txt = (f"{side}: {g}  fingers="
                               f"{''.join('1' if fing[k] else '0' for k in ('thumb','index','middle','ring','pinky'))}"
                               f"  {spd:.2f} m/s ({move})")
                    else:
                        txt = f"{side}: (wrist only)  {spd:.2f} m/s ({move})"
                    cv2.putText(fp_img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, HAND_COLORS[side], 2)
                    y -= 24
                prof.mark("render")

                if show_third:
                    cv2.imshow("Third person (webcam)", frame)
                cv2.imshow("First person", fp_img)

                key = cv2.waitKey(1) & 0xFF
                prof.mark("show")
                if key == ord("q"):
                    break
                if key == ord("h") and hand_landmarker is not None:
                    fingers_on = not fingers_on
                if key == ord("f") and face_landmarker is not None:
                    face_on = not face_on
                if key == ord("s"):
                    show_skel = not show_skel
                if key == ord("m"):
                    show_mesh = not show_mesh
                if key == ord("p"):
                    show_prof = not show_prof
                if key == ord("r"):
                    smooth_overlay = not smooth_overlay
                if key == ord("w"):
                    show_third = not show_third
                    if not show_third:
                        cv2.destroyWindow("Third person (webcam)")
    finally:
        cam.stop()
        infer.shutdown()
        for lm in (hand_landmarker, face_landmarker):
            if lm is not None:
                lm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()