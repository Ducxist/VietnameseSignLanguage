import numpy as np
import math
import random

# ─────────────────────────────────────────────
# Hằng số landmark
# ─────────────────────────────────────────────
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS            = 21
N_TOTAL_LANDMARKS           = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS * 2  # 67

POSE_LM_LEFT_SHOULDER  = 11
POSE_LM_RIGHT_SHOULDER = 12
POSE_LM_LEFT_ELBOW     = 13
POSE_LM_RIGHT_ELBOW    = 14
POSE_LM_LEFT_WRIST     = 15
POSE_LM_RIGHT_WRIST    = 16

IDX_LH_START = N_UPPER_BODY_POSE_LANDMARKS                      # 25
IDX_LH_END   = IDX_LH_START + N_HAND_LANDMARKS                  # 46
IDX_RH_START = IDX_LH_END                                        # 46
IDX_RH_END   = IDX_RH_START + N_HAND_LANDMARKS                  # 67

# ─────────────────────────────────────────────
# Helper nội bộ
# ─────────────────────────────────────────────

def _to_array(seq) -> np.ndarray:
    """
    Chuyển list-of-flat-arrays hoặc ndarray (T, F) → ndarray (T, N_LANDMARKS, 3).
    Trả về None nếu seq rỗng.
    """
    if seq is None or (hasattr(seq, '__len__') and len(seq) == 0):
        return None
    if isinstance(seq, np.ndarray) and seq.ndim == 3:
        return seq.copy()
    # list hoặc (T, F)
    arr = np.array(seq, dtype=np.float64)  # (T, F)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    T, F = arr.shape
    return arr.reshape(T, N_TOTAL_LANDMARKS, 3)


def _from_array(pts: np.ndarray):
    """
    Chuyển (T, N_LANDMARKS, 3) → list of flat arrays  [(F,), (F,), ...]
    để tương thích với code cũ nếu cần.
    """
    T = pts.shape[0]
    flat = pts.reshape(T, -1)
    return [flat[i] for i in range(T)]


def _valid_mask(pts_3d: np.ndarray) -> np.ndarray:
    """
    pts_3d: (..., 3)  → bool mask: True nếu điểm có ít nhất 1 tọa độ != 0
    """
    return np.any(pts_3d != 0, axis=-1)


def _compute_center(pts_3d: np.ndarray) -> tuple:
    """
    pts_3d: (N, 3) – tính median x, y của các điểm hợp lệ.
    Trả về (cx, cy, ok: bool).
    """
    vm = _valid_mask(pts_3d)
    if not np.any(vm):
        return 0.0, 0.0, False
    valid = pts_3d[vm]
    return float(np.median(valid[:, 0])), float(np.median(valid[:, 1])), True


# ─────────────────────────────────────────────
# 1. Scale
# ─────────────────────────────────────────────

def scale_keypoints_sequence(
    keypoints_sequence,
    scale_factor_range=(0.7, 1.26),
    normalize_to_01=True,
):
   
    pts = _to_array(keypoints_sequence)   # (T, N, 3)
    if pts is None:
        return keypoints_sequence

    T = pts.shape[0]
    scale = random.uniform(*scale_factor_range)
    if scale <= 0:
        scale = 1.0

    orig = pts.copy()

    for t in range(T):
        frame = pts[t]                         # (N, 3)
        vm = _valid_mask(frame)
        if not np.any(vm):
            continue
        cx = float(np.median(frame[vm, 0]))
        cy = float(np.median(frame[vm, 1]))

        frame[vm, 0] = (frame[vm, 0] - cx) * scale + cx
        frame[vm, 1] = (frame[vm, 1] - cy) * scale + cy

    if normalize_to_01:
        
        vxy_all = np.any(pts[:, :, :2] != 0, axis=2)  # (T, N)
        if np.any(vxy_all):
            xs = pts[:, :, 0][vxy_all]
            ys = pts[:, :, 1][vxy_all]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            cx_bb = (x0 + x1) / 2.0
            cy_bb = (y0 + y1) / 2.0
            span = max(x1 - x0, y1 - y0)
            span = span if span > 1e-7 else None

            for t in range(T):
                frame = pts[t]
                vm2 = np.any(frame[:, :2] != 0, axis=1)
                if not np.any(vm2):
                    continue
                frame[vm2, 0] = (frame[vm2, 0] - cx_bb) / span + 0.5 if span else 0.5
                frame[vm2, 1] = (frame[vm2, 1] - cy_bb) / span + 0.5 if span else 0.5

    # Loại bỏ frame có NaN/Inf (thay bằng frame gốc tương ứng)
    for t in range(T):
        if not np.all(np.isfinite(pts[t])):
            pts[t] = orig[t]

    return _from_array(pts)


# ─────────────────────────────────────────────
# 2. Rotate
# ─────────────────────────────────────────────

def rotate_keypoints_sequence(
    keypoints_sequence,
    angle_degrees_range=(-15.0, 15.0),
):
    """Xoay 2D toàn bộ chuỗi quanh tâm của mỗi frame."""
    pts = _to_array(keypoints_sequence)
    if pts is None:
        return keypoints_sequence

    T = pts.shape[0]
    angle_rad = math.radians(random.uniform(*angle_degrees_range))
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    orig = pts.copy()

    for t in range(T):
        frame = pts[t]
        vm = _valid_mask(frame)
        if not np.any(vm):
            continue
        cx = float(np.median(frame[vm, 0]))
        cy = float(np.median(frame[vm, 1]))

        dx = frame[vm, 0] - cx
        dy = frame[vm, 1] - cy
        frame[vm, 0] = dx * ca - dy * sa + cx
        frame[vm, 1] = dx * sa + dy * ca + cy

        if not np.all(np.isfinite(frame)):
            pts[t] = orig[t]

    return _from_array(pts)


# ─────────────────────────────────────────────
# 3. Translate
# ─────────────────────────────────────────────

def translate_keypoints_sequence(
    keypoints_sequence,
    translate_x_range=(-0.05, 0.05),
    translate_y_range=(-0.05, 0.05),
    clip_to_01=True,
):
    """Dịch chuyển 2D toàn bộ chuỗi."""
    pts = _to_array(keypoints_sequence)
    if pts is None:
        return keypoints_sequence

    dx = random.uniform(*translate_x_range)
    dy = random.uniform(*translate_y_range)

    vm = _valid_mask(pts)   # (T, N)
    pts[vm, 0] += dx
    pts[vm, 1] += dy

    if clip_to_01:
        pts[vm, 0] = np.clip(pts[vm, 0], 0.0, 1.0)
        pts[vm, 1] = np.clip(pts[vm, 1], 0.0, 1.0)

    return _from_array(pts)


# ─────────────────────────────────────────────
# 4. Time Stretch
# ─────────────────────────────────────────────

def time_stretch_keypoints_sequence(
    keypoints_sequence,
    speed_factor_range=(0.8, 1.2),
):
    
    pts = _to_array(keypoints_sequence)
    if pts is None:
        return keypoints_sequence

    T = pts.shape[0]
    speed = random.uniform(*speed_factor_range)
    if speed <= 0:
        return _from_array(pts)

    new_T = max(1, int(round(T / speed)))
    orig_idx = np.round(np.linspace(0, T - 1, new_T)).astype(int)
    orig_idx = np.clip(orig_idx, 0, T - 1)

    return _from_array(pts[orig_idx])


# ─────────────────────────────────────────────
# 5. Inter-hand Distance  (IK-based)
# ─────────────────────────────────────────────

def _solve_2link_ik(shoulder, wrist_target, l1, l2, orig_elbow=None, orig_wrist=None):
   
    d = np.linalg.norm(wrist_target - shoulder)
    l1 = max(1e-5, l1)
    l2 = max(1e-5, l2)

    if d > l1 + l2 - 1e-5:          # Duỗi thẳng
        if d < 1e-9:
            return shoulder + np.array([l1, 0.0])
        return shoulder + (wrist_target - shoulder) / d * l1

    if d < abs(l1 - l2) + 1e-5:     # Quá gần
        return orig_elbow.copy() if orig_elbow is not None else shoulder + np.array([l1, 0.0])

    if d < 1e-9:
        d = 1e-9
    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h2 = l1**2 - a**2
    h = np.sqrt(max(0.0, h2))

    sw = wrist_target - shoulder
    p2 = shoulder + a * sw / d
    perp = np.array([-sw[1], sw[0]]) / d

    sol1 = p2 + h * perp
    sol2 = p2 - h * perp

    if orig_elbow is None or orig_wrist is None:
        return sol1

    def _side(s, w, e):
        return (w[0] - s[0]) * (e[1] - s[1]) - (w[1] - s[1]) * (e[0] - s[0])

    orig_side = _side(shoulder, orig_wrist, orig_elbow)
    side1 = _side(shoulder, wrist_target, sol1)
    side2 = _side(shoulder, wrist_target, sol2)

    if abs(orig_side) < 1e-3:
        return sol1 if np.linalg.norm(sol1 - orig_elbow) <= np.linalg.norm(sol2 - orig_elbow) else sol2
    if np.sign(side1) == np.sign(orig_side):
        return sol1
    if np.sign(side2) == np.sign(orig_side):
        return sol2
    return sol1 if np.linalg.norm(sol1 - orig_elbow) <= np.linalg.norm(sol2 - orig_elbow) else sol2


def inter_hand_distance(
    keypoints_sequence,
    total_dx_change_range=(-0.1, 0.1),
    overall_dy_shift_range=(-0.03, 0.03),
    clip_to_01=True,
):
    
    pts = _to_array(keypoints_sequence)
    if pts is None:
        return keypoints_sequence

    dx_change = random.uniform(*total_dx_change_range)
    dy_shift  = random.uniform(*overall_dy_shift_range)

    for t in range(pts.shape[0]):
        frame = pts[t]   # (N, 3)

        s_l = frame[POSE_LM_LEFT_SHOULDER,  :2].copy()
        e_l = frame[POSE_LM_LEFT_ELBOW,     :2].copy()
        w_l = frame[POSE_LM_LEFT_WRIST,     :2].copy()
        s_r = frame[POSE_LM_RIGHT_SHOULDER, :2].copy()
        e_r = frame[POSE_LM_RIGHT_ELBOW,    :2].copy()
        w_r = frame[POSE_LM_RIGHT_WRIST,    :2].copy()

        
        l_valid = np.any(s_l != 0) and np.any(e_l != 0) and np.any(w_l != 0)
        r_valid = np.any(s_r != 0) and np.any(e_r != 0) and np.any(w_r != 0)

        # Tính target x cho cổ tay
        if np.any(w_l != 0) and np.any(w_r != 0):
            mid_x = (w_l[0] + w_r[0]) / 2
            dist  = abs(w_r[0] - w_l[0])
            target_dist = max(0.01, dist + dx_change)
            if w_l[0] <= w_r[0]:
                w_l_tx = mid_x - target_dist / 2
                w_r_tx = mid_x + target_dist / 2
            else:
                w_r_tx = mid_x - target_dist / 2
                w_l_tx = mid_x + target_dist / 2
        else:
            w_l_tx, w_r_tx = w_l[0], w_r[0]

        # Tay trái
        if l_valid:
            l1 = np.linalg.norm(e_l - s_l)
            l2 = np.linalg.norm(w_l - e_l)
            w_l_tgt = np.array([w_l_tx, w_l[1]])
            e_l_new = _solve_2link_ik(s_l, w_l_tgt, l1, l2, e_l, w_l)
            if e_l_new is not None:
                dxl = w_l_tx - w_l[0]
                frame[POSE_LM_LEFT_ELBOW, :2] = e_l_new
                frame[POSE_LM_LEFT_WRIST, :2] = w_l_tgt
                vm_lh = _valid_mask(frame[IDX_LH_START:IDX_LH_END])
                frame[IDX_LH_START:IDX_LH_END][vm_lh, 0] += dxl

        # Tay phải
        if r_valid:
            l1 = np.linalg.norm(e_r - s_r)
            l2 = np.linalg.norm(w_r - e_r)
            w_r_tgt = np.array([w_r_tx, w_r[1]])
            e_r_new = _solve_2link_ik(s_r, w_r_tgt, l1, l2, e_r, w_r)
            if e_r_new is not None:
                dxr = w_r_tx - w_r[0]
                frame[POSE_LM_RIGHT_ELBOW, :2] = e_r_new
                frame[POSE_LM_RIGHT_WRIST, :2] = w_r_tgt
                vm_rh = _valid_mask(frame[IDX_RH_START:IDX_RH_END])
                frame[IDX_RH_START:IDX_RH_END][vm_rh, 0] += dxr

        # Shift Y chung
        if abs(dy_shift) > 1e-5:
            arm_idx = [POSE_LM_LEFT_ELBOW, POSE_LM_LEFT_WRIST,
                       POSE_LM_RIGHT_ELBOW, POSE_LM_RIGHT_WRIST]
            for idx in arm_idx:
                if np.any(frame[idx, :2] != 0):
                    frame[idx, 1] += dy_shift
            for idx in range(IDX_LH_START, IDX_RH_END):
                if np.any(frame[idx, :2] != 0):
                    frame[idx, 1] += dy_shift

        # Clip
        if clip_to_01:
            vm = _valid_mask(frame[POSE_LM_LEFT_SHOULDER:])
            tmp = frame[POSE_LM_LEFT_SHOULDER:]
            tmp[vm, 0] = np.clip(tmp[vm, 0], 0.0, 1.0)
            tmp[vm, 1] = np.clip(tmp[vm, 1], 0.0, 1.0)

    return _from_array(pts)


# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
solve_2_link_ik_2d_v2 = _solve_2link_ik
