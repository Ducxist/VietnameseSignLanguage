import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
import json
import os
import time
import threading
import math
from collections import deque
from scipy.interpolate import interp1d
from PIL import ImageFont, ImageDraw, Image

# ─────────────────────────────────────────────
# CONSTANTS — khớp create_data_augment_fixed3.py
# ─────────────────────────────────────────────
SEQUENCE_LEN = 60
N_FEATURES   = 201
L2_REG       = 5e-4
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS            = 21
POSE_LM_LEFT_SHOULDER  = 11
POSE_LM_RIGHT_SHOULDER = 12

MODEL_PATH   = 'final_model.keras'
TFLITE_PATH  = 'final_model.tflite'   # tự sinh lần chạy đầu nếu chưa có
LABEL_PATH   = 'label_map.json'


# ─────────────────────────────────────────────
# [B1] Chuẩn hoá tọa độ theo cơ thể — PHẢI GIỐNG HỆT hàm normalize_body()
# trong augment_function.py (dùng lúc tạo data train). Center theo trung
# điểm 2 vai + scale theo độ rộng vai (median cả chuỗi) -> bất biến với
# vị trí đứng trong khung hình và khoảng cách tới camera.
# ─────────────────────────────────────────────
def normalize_body(keypoints_sequence, eps=1e-6):
    if not keypoints_sequence:
        return keypoints_sequence

    pts = np.array(keypoints_sequence, dtype=np.float64).reshape(-1, N_FEATURES // 3, 3)
    T = pts.shape[0]
    orig = pts.copy()

    shoulder_widths = []
    for t in range(T):
        sl = pts[t, POSE_LM_LEFT_SHOULDER, :2]
        sr = pts[t, POSE_LM_RIGHT_SHOULDER, :2]
        if np.any(sl != 0) and np.any(sr != 0):
            w = float(np.linalg.norm(sl - sr))
            if w > eps:
                shoulder_widths.append(w)

    if shoulder_widths:
        scale = float(np.median(shoulder_widths))
    else:
        vxy_all = np.any(pts[:, :, :2] != 0, axis=2)
        if np.any(vxy_all):
            xs = pts[:, :, 0][vxy_all]
            ys = pts[:, :, 1][vxy_all]
            scale = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        else:
            scale = 1.0
    scale = scale if scale > eps else 1.0

    for t in range(T):
        frame = pts[t]
        sl = frame[POSE_LM_LEFT_SHOULDER, :2]
        sr = frame[POSE_LM_RIGHT_SHOULDER, :2]

        if np.any(sl != 0) and np.any(sr != 0):
            cx, cy = (sl + sr) / 2.0
        else:
            vm_frame = np.any(frame != 0, axis=-1)
            if not np.any(vm_frame):
                continue
            cx = float(np.median(frame[vm_frame, 0]))
            cy = float(np.median(frame[vm_frame, 1]))

        vm = np.any(frame != 0, axis=-1)
        if not np.any(vm):
            continue

        frame[vm, 0] = (frame[vm, 0] - cx) / scale
        frame[vm, 1] = (frame[vm, 1] - cy) / scale

        if not np.all(np.isfinite(frame)):
            pts[t] = orig[t]

    flat = pts.reshape(T, -1)
    return [flat[i] for i in range(T)]

# Tham số phát hiện cử chỉ
MAX_RAW_FRAMES   = 80
MIN_RAW_FRAMES   = 6
HAND_LOST_GRACE  = 4
CONF_THRESHOLD   = 0.75

# ── [SPEED FIX] Cắt cử chỉ theo THỜI GIAN THỰC thay vì số frame.
# Bản cũ dùng MAX_RAW_FRAMES=80 làm mốc ép predict cuối cùng, nhưng nếu
# Holistic xử lý ~250-300ms/frame trên CPU thì 80 frame = tới ~20 giây
# mới bị ép chốt, dù bạn đã ký xong từ lâu. Đổi sang giới hạn theo giây
# thực tế để độ trễ ổn định bất kể tốc độ máy.
MAX_GESTURE_SECONDS = 2.2   # quá thời gian này mà chưa dừng/predict -> ép chốt

# Tham số lọc nhiễu chuyển động (bỏ qua "cử chỉ" mà tay gần như đứng yên)
MOTION_MIN_TOTAL = 0.015   # tổng độ dịch chuyển cổ tay (đơn vị chuẩn hoá 0-1) tối thiểu / cử chỉ ngắn
MOTION_CHECK_MAX_FRAMES = 12  # chỉ áp dụng kiểm tra này nếu cử chỉ tương đối ngắn

# Nội suy khi mất landmark (occlusion) - giữ tối đa bao nhiêu frame liên tiếp
OCCLUSION_CARRY_MAX = 3

# Phát hiện khoảng dừng ngắn giữa 2 từ (KHÔNG cần tay rời khung hình mới cắt cử chỉ)
# [SPEED FIX] Giảm PAUSE_WINDOW và nới ngưỡng để phát hiện dừng nhanh hơn
# -> cắt cử chỉ sớm hơn ngay khi tay chững lại, không phải chờ lâu.
PAUSE_WINDOW = 5                # cân bằng giữa tốc độ (4) và độ ổn định (6 gốc)
PAUSE_MOTION_THRESHOLD = 0.007  # cân bằng giữa 0.006 (gốc) và 0.008 (nhanh)
PAUSE_MIN_GESTURE_FRAMES = MIN_RAW_FRAMES + PAUSE_WINDOW  # phải ký đủ lâu mới bắt đầu xét dừng

# ── Early-commit: predict LIÊN TỤC trong lúc đang ký hiệu, chốt từ ngay
# khi có độ tin cậy đủ cao — không cần chờ phát hiện dừng/mất tay. Đây là
# cách các app "nhận diện trong ~1s" đang làm: đưa nhiều phương án trong
# lúc ký hiệu, phương án nào đạt ngưỡng tin cậy cao thì chốt luôn.
# [SPEED FIX] Hạ mốc bắt đầu thử + thử thường xuyên hơn + hạ nhẹ ngưỡng
# tin cậy để chốt từ sớm hơn (vẫn đủ cao để không tăng nhận nhầm nhiều).
EARLY_COMMIT_ENABLED    = True
EARLY_COMMIT_MIN_FRAMES = 10     # đưa lại về 10 — cần đủ frame mới đáng tin
EARLY_COMMIT_EVERY      = 3      # thử đoán vừa phải, không quá dày
EARLY_COMMIT_CONF       = 0.94   # NÂNG cao hơn cả bản gốc (0.92) -> chỉ chốt sớm khi RẤT chắc

# Độ phân giải đưa vào MediaPipe (khác với độ phân giải hiển thị).
# [FIX #3] Giữ đúng tỉ lệ 4:3 của camera (640x480) — KHÔNG resize thẳng
# về hình vuông như bản cũ (256x256), vì điều đó làm méo toạ độ x/y
# chuẩn hoá mà MediaPipe trả về so với lúc trích xuất data train (video
# được xử lý ở tỉ lệ gốc, không bị bóp méo).
PROCESS_WIDTH, PROCESS_HEIGHT = 256, 192

# True = predict 2 lần/cử chỉ (ổn định hơn nhưng chậm hơn) | False = 1 lần (nhanh nhất)
USE_MAJORITY_VOTE = False

# ─────────────────────────────────────────────
# CUSTOM OBJECTS
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package='Custom')
class TemporalAttention(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        D = input_shape[-1]
        self.W_a = self.add_weight(name='W_a', shape=(D, D), initializer='glorot_uniform', regularizer=l2(L2_REG), trainable=True)
        self.v_a = self.add_weight(name='v_a', shape=(D, 1), initializer='glorot_uniform', regularizer=l2(L2_REG), trainable=True)
        super().build(input_shape)
    def call(self, hidden_states):
        score = tf.nn.tanh(tf.matmul(hidden_states, self.W_a))
        score = tf.squeeze(tf.matmul(score, self.v_a), axis=-1)
        alpha = tf.nn.softmax(score, axis=-1)
        alpha = tf.expand_dims(alpha, axis=-1)
        return tf.reduce_sum(hidden_states * alpha, axis=1)
    def get_config(self): return super().get_config()

@tf.keras.utils.register_keras_serializable(package='Custom')
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, min_lr, warmup_steps, total_steps, **kwargs):
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + tf.math.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    def get_config(self):
        return {'base_lr': self.base_lr, 'min_lr': self.min_lr, 'warmup_steps': self.warmup_steps, 'total_steps': self.total_steps}

CUSTOM_OBJECTS = {
    'TemporalAttention': TemporalAttention,
    'WarmupCosineDecay': WarmupCosineDecay,
}

# ─────────────────────────────────────────────
# NỘI SUY VỀ SEQUENCE_LEN — khớp pipeline train
# ─────────────────────────────────────────────
def interpolate_keypoints(seq: list, target_len: int):
    if not seq:
        return None
    arr = np.stack(seq, axis=0).astype(np.float32)
    T, F = arr.shape
    if T == target_len:
        return arr
    if T == 1:
        return np.repeat(arr, target_len, axis=0)
    kind = 'cubic' if T >= 4 else 'linear'
    t_orig = np.linspace(0, 1, T)
    t_target = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, F), dtype=np.float32)
    for fi in range(F):
        f = interp1d(t_orig, arr[:, fi], kind=kind, bounds_error=False, fill_value='extrapolate')
        result[:, fi] = f(t_target)
    return result


# ─────────────────────────────────────────────
# TFLITE: convert 1 lần + load interpreter
# ─────────────────────────────────────────────
def ensure_tflite(model_path: str, tflite_path: str) -> str:
    if os.path.exists(tflite_path):
        return tflite_path

    keras_model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # [A1] Float16 quantization: giảm kích thước model ~2x và tăng tốc suy
    # luận trên CPU, độ chính xác gần như không đổi (khác hẳn INT8 - loại
    # đó cần representative dataset và có thể làm giảm accuracy).
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    try:
        tflite_model = converter.convert()
    except Exception:
        # Một số layer (vd LSTM) cần SELECT_TF_OPS để convert được
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_path


# ─────────────────────────────────────────────
# ASYNCHRONOUS PROCESSOR
# ─────────────────────────────────────────────
class SignLanguageProcessor(threading.Thread):
    def __init__(self, model_path, tflite_path, label_path):
        threading.Thread.__init__(self)
        self.daemon = True
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.label_path = label_path
        self.frame_to_process = None
        self.running = True

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.keras_model = None
        self.use_tflite = False
        self.inv_label_map = None

        self.lock = threading.Lock()
        self.prediction_info = {"word": "Đang khởi tạo...", "conf": 0.0}
        self.sentence = []
        self.fps_info = {"capture": 0.0, "process": 0.0}

        self.raw_buffer = []
        self.gesture_active = False
        self.gesture_start_time = None   # [SPEED FIX] mốc thời gian bắt đầu cử chỉ (giây thực)
        self.hand_lost_count = 0
        self._recent_motions = deque(maxlen=PAUSE_WINDOW)

        # Occlusion carry-forward state
        self._last_lhand = None
        self._last_rhand = None
        self._lhand_missing_streak = 0
        self._rhand_missing_streak = 0

    # ---------- Trích xuất + nội suy occlusion ----------
    def extract_keypoints(self, results) -> np.ndarray:
        pose  = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3), dtype=np.float32)
        lhand = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)
        rhand = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark[:N_UPPER_BODY_POSE_LANDMARKS]):
                pose[i] = [lm.x, lm.y, lm.z]

        has_l = bool(results.left_hand_landmarks)
        has_r = bool(results.right_hand_landmarks)

        if has_l:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                lhand[i] = [lm.x, lm.y, lm.z]
            self._last_lhand = lhand.copy()
            self._lhand_missing_streak = 0
        else:
            if self._last_lhand is not None and self._lhand_missing_streak < OCCLUSION_CARRY_MAX:
                lhand = self._last_lhand.copy()
                self._lhand_missing_streak += 1
            else:
                self._last_lhand = None

        if has_r:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                rhand[i] = [lm.x, lm.y, lm.z]
            self._last_rhand = rhand.copy()
            self._rhand_missing_streak = 0
        else:
            if self._last_rhand is not None and self._rhand_missing_streak < OCCLUSION_CARRY_MAX:
                rhand = self._last_rhand.copy()
                self._rhand_missing_streak += 1
            else:
                self._last_rhand = None

        kp = np.concatenate([pose, lhand, rhand]).flatten()
        return kp, has_l, has_r

    # ---------- Lọc nhiễu chuyển động ----------
    @staticmethod
    def _motion_energy(buf):
        """Tổng độ dịch chuyển cổ tay trái+phải qua các frame (dùng để loại cử chỉ 'đứng yên')."""
        if len(buf) < 2:
            return 0.0
        # index cổ tay trong vector 201: pose wrist trái=15, phải=16 (mỗi landmark 3 giá trị)
        idx_l = 15 * 3
        idx_r = 16 * 3
        arr = np.stack(buf, axis=0)
        wl = arr[:, idx_l:idx_l + 2]
        wr = arr[:, idx_r:idx_r + 2]
        motion = np.sum(np.linalg.norm(np.diff(wl, axis=0), axis=1)) + \
                 np.sum(np.linalg.norm(np.diff(wr, axis=0), axis=1))
        return float(motion)

    @staticmethod
    def _frame_motion(kp_prev, kp_curr):
        """Độ dịch chuyển cổ tay trái+phải giữa 2 frame liên tiếp (dùng để phát hiện pause)."""
        idx_l = 15 * 3
        idx_r = 16 * 3
        dl = np.linalg.norm(kp_curr[idx_l:idx_l + 2] - kp_prev[idx_l:idx_l + 2])
        dr = np.linalg.norm(kp_curr[idx_r:idx_r + 2] - kp_prev[idx_r:idx_r + 2])
        return float(dl + dr)

    # ---------- Inference qua TFLite hoặc Keras (fallback) ----------
    def _infer(self, input_data: np.ndarray) -> np.ndarray:
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(np.float32))
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            return self.keras_model(input_data, training=False).numpy()[0]

    def _predict_probs(self, buf) -> np.ndarray:
        """Chạy model trên buf (list keypoint), trả về vector xác suất trung bình
        (majority-vote nhẹ nếu bật). Dùng chung cho early-commit lẫn predict cuối."""
        variants = [buf]
        if USE_MAJORITY_VOTE and len(buf) >= MIN_RAW_FRAMES + 4:
            variants.append(buf[2:-2])

        probs_sum = None
        for v in variants:
            # [B1] Chuẩn hoá theo cơ thể TRƯỚC khi nội suy - phải khớp
            # đúng với bước normalize_body() đã áp dụng lúc tạo data train
            # (create_data_augment_fixed3.py), nếu không sẽ lại lệch pha
            # train/inference như các lỗi trước.
            v_norm = normalize_body(v)
            interp = interpolate_keypoints(v_norm, SEQUENCE_LEN)
            input_data = np.expand_dims(interp, axis=0)
            res = self._infer(input_data)
            probs_sum = res if probs_sum is None else probs_sum + res

        return probs_sum / len(variants)

    def _commit_word(self, idx: int, conf: float):
        """Ghi nhận 1 từ vào câu (tránh lặp liên tiếp cùng 1 từ)."""
        word = self.inv_label_map[idx]
        self.prediction_info = {"word": word, "conf": conf}
        if not self.sentence or word != self.sentence[-1]:
            self.sentence.append(word)
            if len(self.sentence) > 7:
                self.sentence.pop(0)

    # ---------- Early-commit: thử đoán sớm trong lúc đang ký hiệu ----------
    def _try_early_commit(self) -> bool:
        """Predict thử trên buffer HIỆN TẠI (chưa kết thúc cử chỉ). Nếu độ
        tin cậy vượt ngưỡng rất cao (EARLY_COMMIT_CONF) thì chốt từ NGAY,
        không cần chờ phát hiện dừng/mất tay -> giảm độ trễ đáng kể.
        Trả về True nếu đã chốt (buffer sẽ được reset để bắt đầu từ tiếp theo)."""
        buf = self.raw_buffer
        probs = self._predict_probs(buf)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf >= EARLY_COMMIT_CONF:
            with self.lock:
                self.fps_info["process"] = 0.0  # early-commit không tính vào process_ms hiển thị
                self._commit_word(idx, conf)
            # Reset trạng thái cử chỉ để bắt đầu thu thập từ TIẾP THEO,
            # tránh gộp lẫn frame của từ vừa chốt vào từ sau.
            self.raw_buffer = []
            self.gesture_active = True   # tay vẫn có thể đang trong khung -> giữ active
            self.gesture_start_time = time.time()   # [SPEED FIX] reset mốc thời gian cho từ tiếp theo
            self.hand_lost_count = 0
            self._recent_motions.clear()
            return True
        return False

    def _predict_and_reset(self):
        buf = self.raw_buffer
        self.raw_buffer = []
        self.gesture_active = False
        self.gesture_start_time = None   # [SPEED FIX]
        self.hand_lost_count = 0
        self._recent_motions.clear()

        if len(buf) < MIN_RAW_FRAMES:
            with self.lock:
                self.prediction_info = {"word": "Chờ ký hiệu", "conf": 0.0}
            return

        # Lọc nhiễu: cử chỉ ngắn mà gần như không di chuyển -> bỏ qua, không đoán bừa
        if len(buf) <= MOTION_CHECK_MAX_FRAMES:
            energy = self._motion_energy(buf)
            if energy < MOTION_MIN_TOTAL:
                with self.lock:
                    self.prediction_info = {"word": "Chờ ký hiệu", "conf": 0.0}
                return

        t0 = time.time()
        probs_avg = self._predict_probs(buf)
        idx = int(np.argmax(probs_avg))
        conf = float(probs_avg[idx])
        process_ms = (time.time() - t0) * 1000.0

        with self.lock:
            self.fps_info["process"] = process_ms
            if conf > CONF_THRESHOLD:
                self._commit_word(idx, conf)
            else:
                self.prediction_info = {"word": "...", "conf": conf}

    def run(self):
        try:
            # Ưu tiên thử TFLite (nhanh hơn). Nếu môi trường không hỗ trợ
            # (thường do model cần "Select TensorFlow ops" mà interpreter
            # không có Flex delegate) -> tự động rơi về chạy bằng Keras.
            try:
                tflite_path = ensure_tflite(self.model_path, self.tflite_path)
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()
                # Thử chạy 1 lần với input giả để chắc chắn interpreter hoạt động
                inp_details = interpreter.get_input_details()
                dummy = np.zeros(inp_details[0]['shape'], dtype=np.float32)
                interpreter.set_tensor(inp_details[0]['index'], dummy)
                interpreter.invoke()

                self.interpreter = interpreter
                self.input_details = inp_details
                self.output_details = interpreter.get_output_details()
                self.use_tflite = True
            except Exception as tflite_err:
                print(f"⚠️ TFLite không dùng được ({str(tflite_err)[:80]}...) -> chạy bằng Keras model.")
                self.keras_model = tf.keras.models.load_model(
                    self.model_path, custom_objects=CUSTOM_OBJECTS, compile=False
                )
                self.use_tflite = False

            with open(self.label_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            self.inv_label_map = {v: k for k, v in label_map.items()}

            with self.lock:
                self.prediction_info["word"] = "Sẵn sàng" + (" (TFLite)" if self.use_tflite else " (Keras)")
        except Exception as e:
            with self.lock:
                self.prediction_info["word"] = f"Lỗi Load: {str(e)[:40]}"
            return

        # [FIX #2] Dùng ĐÚNG Holistic giống hệt lúc trích xuất data train
        # (create_data_augment_fixed3.py) thay vì 2 model Pose+Hands tách
        # rời — 2 pipeline này cho toạ độ/khả năng phát hiện tay khác nhau,
        # gây lệch phân bố đặc trưng so với lúc train.
        mp_holistic = mp.solutions.holistic
        holistic_model = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            while self.running:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None
                    t0 = time.time()

                    # [FIX #3] Resize giữ NGUYÊN tỉ lệ khung hình 4:3 của
                    # camera (không resize thẳng về hình vuông như trước,
                    # vì việc đó làm méo toạ độ x/y chuẩn hoá so với lúc
                    # train, khi frame video được xử lý ở tỉ lệ gốc).
                    small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
                    image_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False

                    results = holistic_model.process(image_rgb)

                    kp, has_l, has_r = self.extract_keypoints(results)
                    has_hand = has_l or has_r

                    if has_hand:
                        if not self.gesture_active:
                            # [SPEED FIX] Bắt đầu cử chỉ mới -> ghi mốc thời gian thực
                            self.gesture_start_time = time.time()
                        self.gesture_active = True
                        self.hand_lost_count = 0
                        self.raw_buffer.append(kp)
                        with self.lock:
                            self.prediction_info["word"] = "Đang ký hiệu..."

                        if len(self.raw_buffer) >= 2:
                            motion = self._frame_motion(self.raw_buffer[-2], self.raw_buffer[-1])
                            self._recent_motions.append(motion)

                        # ── Early-commit: thử đoán sớm định kỳ trong lúc đang
                        # ký hiệu, chốt NGAY nếu đã rất tự tin -> không cần
                        # chờ đủ điều kiện phát hiện dừng bên dưới nữa.
                        committed_early = False
                        if (EARLY_COMMIT_ENABLED
                                and len(self.raw_buffer) >= EARLY_COMMIT_MIN_FRAMES
                                and len(self.raw_buffer) % EARLY_COMMIT_EVERY == 0):
                            committed_early = self._try_early_commit()

                        # Cắt cử chỉ khi phát hiện khoảng dừng ngắn -> KHÔNG cần đợi
                        # tay rời khỏi khung hình mới predict (đây là lý do các
                        # từ sau từ đầu tiên bị trễ rất lâu ở bản trước).
                        if not committed_early:
                            is_pausing = (
                                len(self.raw_buffer) >= PAUSE_MIN_GESTURE_FRAMES
                                and len(self._recent_motions) == PAUSE_WINDOW
                                and (sum(self._recent_motions) / PAUSE_WINDOW) < PAUSE_MOTION_THRESHOLD
                            )
                            # [SPEED FIX] Ép predict theo THỜI GIAN THỰC (giây) thay
                            # vì theo số frame — bất biến với tốc độ xử lý của máy.
                            gesture_elapsed = (
                                (time.time() - self.gesture_start_time)
                                if self.gesture_start_time is not None else 0.0
                            )
                            timed_out = gesture_elapsed >= MAX_GESTURE_SECONDS
                            if is_pausing or timed_out or len(self.raw_buffer) >= MAX_RAW_FRAMES:
                                self._predict_and_reset()
                    elif self.gesture_active:
                        self.hand_lost_count += 1
                        self.raw_buffer.append(kp)
                        if self.hand_lost_count >= HAND_LOST_GRACE:
                            self._predict_and_reset()
                    else:
                        with self.lock:
                            self.prediction_info = {"word": "Chờ ký hiệu", "conf": 0.0}

                    holistic_ms = (time.time() - t0) * 1000.0
                    with self.lock:
                        self.fps_info["capture"] = holistic_ms
                else:
                    time.sleep(0.005)
        finally:
            holistic_model.close()

    def get_state(self):
        with self.lock:
            return dict(self.prediction_info), list(self.sentence), dict(self.fps_info)

    # ---------- Chỉnh sửa câu bằng phím (gọi từ vòng lặp hiển thị ở main()) ----------
    def delete_last_word(self):
        """Xoá từ cuối cùng khỏi câu đang hiển thị (Backspace)."""
        with self.lock:
            if self.sentence:
                self.sentence.pop()

    def clear_sentence(self):
        """Xoá toàn bộ câu đang hiển thị (phím C)."""
        with self.lock:
            self.sentence.clear()


# ─────────────────────────────────────────────
# UI & VIETNAMESE SUPPORT
# ─────────────────────────────────────────────
def draw_ui(img, word, conf, sentence_list, fps_display, fps_info):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font = None
    font_large = font_small = font_tiny = None
    for path in font_paths:
        try:
            font_large = ImageFont.truetype(path, 28)
            font_small = ImageFont.truetype(path, 22)
            font_tiny = ImageFont.truetype(path, 16)
            font = True
            break
        except Exception:
            continue
    if not font:
        font_large = font_small = font_tiny = ImageFont.load_default()

    draw.rectangle([0, 0, 640, 95], fill=(30, 30, 30))
    draw.text((15, 10), f"Nhận diện: {word} ({conf*100:.0f}%)", font=font_large, fill=(255, 255, 0))

    # ── Câu nhận diện: nối liền các từ thành 1 đoạn văn bản, tự xuống dòng ──
    sentence_str = " ".join(sentence_list)
    box_left, box_right = 15, 625
    max_text_width = box_right - box_left
    max_lines = 3
    line_height = 26

    lines = _wrap_text_to_lines(draw, sentence_str, font_small, max_text_width, max_lines)

    box_height = 34 + line_height * max(1, len(lines))
    box_top = 480 - box_height
    draw.rectangle([0, box_top, 640, 480], fill=(20, 20, 20))
    draw.text((box_left, box_top + 8), "Câu:", font=font_small, fill=(150, 220, 150))

    text_y = box_top + 8
    for line in lines:
        draw.text((box_left + 60, text_y), line, font=font_small, fill=(0, 255, 0))
        text_y += line_height

    # Overlay FPS / thời gian xử lý
    fps_text = f"Camera FPS: {fps_display:.0f}  |  MediaPipe: {fps_info.get('capture', 0):.0f}ms  |  Predict: {fps_info.get('process', 0):.0f}ms"
    draw.text((15, 55), fps_text, font=font_tiny, fill=(200, 200, 200))

    # Gợi ý phím tắt chỉnh sửa câu
    hint_text = "[Backspace] Xoá từ cuối   |   [C] Xoá cả câu   |   [Q] Thoát"
    draw.text((15, 75), hint_text, font=font_tiny, fill=(150, 150, 150))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _wrap_text_to_lines(draw, text, font, max_width, max_lines):
    """Gộp các từ thành đoạn văn bản liền mạch, tự xuống dòng theo max_width.
    Nếu vượt quá max_lines, chỉ giữ lại các dòng CUỐI (câu mới nhất)."""
    if not text:
        return [""]

    words = text.split(" ")
    lines, current = [], ""

    for w in words:
        candidate = w if not current else f"{current} {w}"
        width = draw.textlength(candidate, font=font)
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    processor = SignLanguageProcessor(MODEL_PATH, TFLITE_PATH, LABEL_PATH)
    processor.start()

    # [CAMERA FPS FIX] Nhiều webcam chỉ đạt ~12-15 FPS ở 640x480 vì driver
    # mặc định dùng codec YUYV (không nén, băng thông USB lớn). Ép dùng
    # MJPG (nén) + khai báo rõ FPS mong muốn thường đưa camera lên 30 FPS.
    # Trên Windows nếu vẫn không cải thiện, thử đổi backend:
    #   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📷 Camera thực tế: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f} FPS (driver báo cáo)")

    print("✅ V3 — TFLITE + GIẢM ĐỘ PHÂN GIẢI + LỌC NHIỄU + OCCLUSION + FPS + CUTOFF THEO THỜI GIAN ĐANG CHẠY...")

    prev_t = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [FIX #1] Lật gương giống hệt lúc quay video train
        # (record_dataset_fixed.py flip mọi frame trước khi trích xuất
        # keypoint). Thiếu dòng này khiến toạ độ x bị đối xứng ngược và
        # tay trái/phải bị hoán đổi so với lúc train.
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

        if processor.frame_to_process is None:
            processor.frame_to_process = frame.copy()

        pred_info, sentence, fps_info = processor.get_state()
        display_frame = draw_ui(frame, pred_info['word'], pred_info['conf'], sentence, fps_display, fps_info)

        cv2.imshow('Sign Language Recognition V3', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            processor.running = False
            break
        elif key in (8, 127):          # Backspace (đa số hệ điều hành trả về 8, một số trả về 127) -> xoá từ cuối
            processor.delete_last_word()
        elif key in (ord('c'), ord('C')):  # Xoá toàn bộ câu đang hiển thị
            processor.clear_sentence()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()