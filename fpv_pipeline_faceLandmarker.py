import os
import time
import urllib.request
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pose_landmarker_full.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
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
VIS_MIN = 0.5           # hands visibility threshold
EMA_ALPHA = 0.5
FINGER_EMA = 0.6        # smoothing of finger offsets
HAND_HOLD_S = 0.25      # keep last finger data this long if the hand is lost
MATCH_MAX_DIST = 0.15   # max normalized image distance to pair hand<->wrist
MOVING_SPEED = 0.25     # m/s: wrist speed (relative to head) counted as moving
EYE_HEIGHT = 1.6
HFOV_DEG = 90.0
VIEW_W, VIEW_H = 640, 480

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


# ------------------------------ Core math ----------------------------------
def to_right_handed(lm):
    """MediaPipe world coords (x right, y DOWN, z away from camera, meters)
    -> right-handed frame (x right, y UP, z toward camera)."""
    return np.array([lm.x, -lm.y, -lm.z], dtype=float)


def invert_rigid(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def to_head_frame(points, T):
    """p' = T^-1 p  (x right, y up, z forward, relative to the eyes)."""
    points = np.asarray(points, dtype=float)
    homo = np.hstack([points, np.ones((len(points), 1))])
    return (invert_rigid(T) @ homo.T).T[:, :3]


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
    """1.0 when excess <= 0, then falls off smoothly (no hard reject)."""
    e = max(excess, 0.0) / soft
    return 1.0 / (1.0 + e * e)


def one_euro_alpha(cutoff_hz, dt):
    return 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)


MIRROR_X = np.diag([-1.0, 1.0, 1.0])


# --------------------------- Landmark smoothing ----------------------------
class LandmarkTracker:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.pos = {}
        self.vis = {}
        self.raw = {}

    def update(self, world_landmarks, ids):
        for i in ids:
            lm = world_landmarks[i]
            v = lm.visibility if lm.visibility is not None else 1.0
            self.vis[i] = v
            self.raw[i] = to_right_handed(lm)
            if v < VIS_MIN:
                continue
            p = to_right_handed(lm)
            if i in self.pos:
                self.pos[i] = self.alpha * p + (1 - self.alpha) * self.pos[i]
            else:
                self.pos[i] = p

    def visible(self, ids):
        return all(self.vis.get(i, 0.0) >= VIS_MIN and i in self.pos for i in ids)


# --------------------------- Occlusion measurement -------------------------
def build_hand_mask(groups, w, h, grow_px, scale=4):
    """Soft mask (0..1, low-res) of where hands are in the webcam image.
    groups: list of lists of normalized (x, y) points, one list per hand."""
    import cv2

    mw, mh = max(w // scale, 1), max(h // scale, 1)
    mask = np.zeros((mh, mw), np.uint8)
    for pts in groups:
        if not pts:
            continue
        px = np.array([[x * mw, y * mh] for x, y in pts], dtype=np.float32)
        if len(px) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(px).astype(np.int32), 255)
        for x, y in px:
            cv2.circle(mask, (int(x), int(y)), 1, 255, -1)
    k = int(grow_px / scale) * 2 + 1
    if k > 1:
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        mask = cv2.GaussianBlur(mask, (0, 0), max(k / 4.0, 1.0))
    return np.clip(mask.astype(np.float32) / 255.0 * 1.5, 0.0, 1.0)


def sample_mask(mask, xy):
    """Occlusion (0..1) at normalized image points xy (N,2)."""
    mh, mw = mask.shape
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
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


def analyze_hand(off):
    """off: (21,3) hand points. Returns (gesture, finger_states, pinch_m)."""
    def d(a, b):
        return float(np.linalg.norm(off[a] - off[b]))

    palm = max(d(0, 9), 1e-6)
    fingers = {
        "thumb": d(4, 17) > d(3, 17) * 1.1,
        "index": d(8, 0) > d(6, 0) * 1.05,
        "middle": d(12, 0) > d(10, 0) * 1.05,
        "ring": d(16, 0) > d(14, 0) * 1.05,
        "pinky": d(20, 0) > d(18, 0) * 1.05,
    }
    pinch = d(4, 8)
    ext = [fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]

    if pinch < 0.35 * palm:
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
    """Smooths finger offsets and tracks wrist speed (relative to the head)."""

    def __init__(self):
        self.offsets = {}     # side -> (21,3) smoothed, wrist at origin
        self.last_seen = {}
        self.prev = {}        # side -> (t, wrist_in_head)
        self.speed = {"L": 0.0, "R": 0.0}
        self.info = {}        # side -> (gesture, fingers, pinch)

    def update_fingers(self, side, hand_world, t):
        off = np.array([to_right_handed(l) for l in hand_world])
        off -= off[0]
        if side in self.offsets:
            a = FINGER_EMA
            self.offsets[side] = a * off + (1 - a) * self.offsets[side]
        else:
            self.offsets[side] = off
        self.last_seen[side] = t
        self.info[side] = analyze_hand(self.offsets[side])

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

    def render(self, T, hands):
        """hands: {side: (points_world (N,3), connections, kind)}"""
        import cv2

        img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        img[:] = (60, 40, 30)
        R = T[:3, :3]

        # Ground grid
        gx = np.arange(np.floor(self.cam[0]) - 12, np.floor(self.cam[0]) + 13)
        gz = np.arange(np.floor(self.cam[2]) - 12, np.floor(self.cam[2]) + 13)
        GX, GZ = np.meshgrid(gx, gz)
        ground = np.stack(
            [GX.ravel(), np.full(GX.size, -EYE_HEIGHT), GZ.ravel()], axis=1
        )
        view = (ground - self.cam) @ R
        px, vis, _ = project(view, VIEW_W, VIEW_H)
        for (x, y), ok, dz in zip(px, vis, view[:, 2]):
            if ok:
                cv2.circle(img, (int(x), int(y)),
                           int(np.clip(6 / max(dz, 0.5), 1, 4)),
                           (110, 160, 110), -1)

        # Hands and fingers (real tracked positions, head-relative)
        for side, (pts, conns, kind) in hands.items():
            color = HAND_COLORS[side]
            ph = to_head_frame(pts, T)
            px, vis, f = project(ph, VIEW_W, VIEW_H)
            for a, b in conns:
                if vis[a] and vis[b]:
                    cv2.line(img, tuple(px[a].astype(int)),
                             tuple(px[b].astype(int)), color, 3)
            for j in range(len(ph)):
                if not vis[j]:
                    continue
                tip = kind == "full" and j in FINGERTIPS
                base = 0.010 if kind == "full" else 0.03
                rad = int(np.clip(base * f / ph[j, 2] * (1.4 if tip else 1.0), 2, 25))
                cv2.circle(img, tuple(px[j].astype(int)), rad,
                           (255, 255, 255) if tip else color, -1)

        cv2.drawMarker(img, (VIEW_W // 2, VIEW_H // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 12, 1)
        return img


# ---------------------------------- Setup ----------------------------------
def ensure_model(path, url, label):
    if os.path.exists(path):
        return True
    print(f"Downloading {label} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Done.")
        return True
    except Exception as e:
        print(f"Could not download the {label} ({e}).")
        print(f"Download it manually from:\n  {url}\nand save it as:\n  {path}")
        return False


def conf_color(c):
    """BGR: red (0) -> green (1)."""
    c = float(np.clip(c, 0.0, 1.0))
    return (0, int(255 * c), int(255 * (1.0 - c)))


# ---------------------------------- Main -----------------------------------
def main():
    import cv2
    import mediapipe as mp

    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions

    if not ensure_model(MODEL_PATH, POSE_MODEL_URL, "pose_landmarker_full.task"):
        return  # the pose model is required

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=MODEL_PATH),
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        raise SystemExit

    tracker = LandmarkTracker()
    hands_tr = HandTracker()
    fpv = FirstPersonView()
    head_est = HeadPoseEstimator()
    last_T = None

    start = time.monotonic()
    last_ts_ms = -1
    prev_t = start
    fps = 0.0

    third_lines = [(0, 2), (2, 7), (0, 5), (5, 8),
                   (15, 17), (15, 19), (15, 21),
                   (16, 18), (16, 20), (16, 22)]

    with vision.PoseLandmarker.create_from_options(pose_opts) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read webcam frame.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.monotonic()
            ts_ms = max(int((now - start) * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            result = landmarker.detect_for_video(image, ts_ms)
            hand_res = None
            if fingers_on:
                hand_res = hand_landmarker.detect_for_video(image, ts_ms)
            face_res = None
            if face_on:
                face_res = face_landmarker.detect_for_video(image, ts_ms)

            hands_out = {}
            status = "No pose detected"

            pose_ok = bool(result.pose_world_landmarks and result.pose_landmarks)
            img_lm = None
            if pose_ok:
                world = result.pose_world_landmarks[0]
                img_lm = result.pose_landmarks[0]
                tracker.update(world, USED_IDS)

            # ================= HEAD: measurements + confidence =================
            # Face Landmarker: mesh points + rotation
            face_pts, face_R = None, None
            if face_res is not None and face_res.face_landmarks:
                fl = face_res.face_landmarks[0]
                face_pts = np.array([[fl[i].x, fl[i].y] for i in FACE_KEY])
                mats = face_res.facial_transformation_matrixes
                if mats is not None and len(mats) > 0:
                    face_R = face_rotation_from_matrix(mats[0])

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
            occ_mask = build_hand_mask(hand_groups, w, h, grow_px=0.25 * head_px)

            # face confidence from how much of the face is covered
            c_face, face_occ_pts = 0.0, None
            if face_pts is not None and face_R is not None:
                face_occ_pts = sample_mask(occ_mask, face_pts)
                face_occ = float((face_occ_pts * FACE_KEY_W).sum() / FACE_KEY_W.sum())
                c_face = FACE_BASE_CONF * (1.0 - face_occ) ** FACE_OCC_POWER

            # per-landmark weights for the pose head points
            head_raw, head_w = {}, {}
            if pose_ok:
                for i in HEAD_IDS:
                    occ = float(sample_mask(occ_mask, [[img_lm[i].x, img_lm[i].y]])[0])
                    head_raw[i] = tracker.raw[i]
                    head_w[i] = vis_weight(tracker.vis[i]) * (1.0 - occ) ** 2

            last_T = head_est.update(now, head_raw, head_w, face_R, c_face)
            hinfo = head_est.info
            head_col = conf_color(hinfo["conf"])

            if pose_ok:
                # ---- Third-person overlay (colour = confidence, red -> green) ----
                for a, b in third_lines:
                    cv2.line(frame,
                             (int(img_lm[a].x * w), int(img_lm[a].y * h)),
                             (int(img_lm[b].x * w), int(img_lm[b].y * h)),
                             head_col if a in HEAD_IDS else (0, 255, 0), 2)
                for i in USED_IDS:
                    if i in HEAD_IDS:
                        color = conf_color(head_w.get(i, 0.0))
                    elif i in HAND_IDS:
                        color = (0, 165, 255)
                    else:
                        color = (255, 200, 0)
                    cv2.circle(frame, (int(img_lm[i].x * w), int(img_lm[i].y * h)),
                               5, color, -1)
                status = f"Head {hinfo['mode']}"

                # ---- Fingers: pair detections with pose wrists ----
                if hand_res is not None and hand_res.hand_landmarks:
                    wrist_ok = {s: tracker.visible([g[0]]) for s, g in SIDES}
                    pairs = match_hands(hand_res.hand_landmarks, img_lm, wrist_ok)
                    for side, hi in pairs.items():
                        hands_tr.update_fingers(
                            side, hand_res.hand_world_landmarks[hi], now)
                        col = HAND_COLORS[side]
                        hl = hand_res.hand_landmarks[hi]
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame,
                                     (int(hl[a].x * w), int(hl[a].y * h)),
                                     (int(hl[b].x * w), int(hl[b].y * h)), col, 2)

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

            # face mesh key points: cyan = clear, red = covered by a hand
            if face_pts is not None:
                for k, (fx, fy) in enumerate(face_pts):
                    occd = face_occ_pts is not None and face_occ_pts[k] > 0.5
                    cv2.circle(frame, (int(fx * w), int(fy * h)), 2,
                               (0, 0, 255) if occd else (255, 255, 0), -1)

            # ---- First-person view ----
            if last_T is not None:
                fp_img = fpv.render(last_T, hands_out)
            else:
                fp_img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
                cv2.putText(fp_img, "Waiting for head landmarks...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # ---- HUD ----
            cv2.putText(fp_img,
                        f"HEAD: {hinfo['mode']}  conf={hinfo['conf']:.2f} "
                        f"face={hinfo['c_face']:.2f} pose={hinfo['c_pose']:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_col, 2)
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

            cv2.putText(frame,
                        f"{status} | FPS {fps:.0f} | fingers {'ON' if fingers_on else 'OFF'}"
                        f" | face {'ON' if face_on else 'OFF'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Third person (webcam)", frame)
            cv2.imshow("First person", fp_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h") and hand_landmarker is not None:
                fingers_on = not fingers_on
            if key == ord("f") and face_landmarker is not None:
                face_on = not face_on

    for lm in (hand_landmarker, face_landmarker):
        if lm is not None:
            lm.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
