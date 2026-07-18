import cv2
import numpy as np
import os
import csv
import json
import random
import logging
import argparse
import multiprocessing
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime

import mediapipe as mp
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

from augment_function import (
    inter_hand_distance,
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
)

# =============================================
# Hang so landmark
# =============================================
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS            = 21
N_TOTAL_LANDMARKS           = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS * 2  # 67
N_FEATURES                  = N_TOTAL_LANDMARKS * 3  

# =============================================
# Danh sach ham tang cuong
# =============================================
AUGMENTATIONS = [
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
    inter_hand_distance,
]


@dataclass
class Config:
    dataset_path: str
    video_folder: str
    label_file: str
    data_path: str
    log_path: str
    sequence_length: int
    max_raw_frames: int
    num_aug_samples: int
    max_augs_per_sample: int
    model_complexity: int
    workers: int
    resume: bool
    limit: int | None


# =============================================
# LOGGING
# =============================================

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "create_data_augment.log")

    logger = logging.getLogger("create_data_augment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


class Timer:
    def __init__(self):
        self._start = datetime.now()

    def elapsed(self):
        return datetime.now() - self._start


# =============================================
# Trich xuat keypoints
# =============================================

def extract_keypoints(results) -> np.ndarray:
    """Tra ve flat array (N_TOTAL * 3,)."""
    pose  = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3), dtype=np.float32)
    lhand = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)
    rhand = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:N_UPPER_BODY_POSE_LANDMARKS]):
            pose[i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lhand[i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rhand[i] = [lm.x, lm.y, lm.z]

    return np.concatenate([pose, lhand, rhand]).flatten()


def _safe_extract_from_frame(frame, holistic):
    """Trich xuat keypoint an toan tu 1 frame BGR, tra ve None neu loi."""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        return extract_keypoints(results)
    except Exception:
        return None


def read_video_keypoints(video_path: str, holistic, max_raw_frames: int) -> list:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0:
        step = max(1, total // max_raw_frames)
        target_indices = set(range(0, total, step))

        keypoints_list = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_indices:
                kp = _safe_extract_from_frame(frame, holistic)
                if kp is not None:
                    keypoints_list.append(kp)
            frame_idx += 1

        cap.release()
        return keypoints_list

    # [FIX] Metadata frame count khong tin cay -> doc het video, lay mau deu sau
    all_kp = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = _safe_extract_from_frame(frame, holistic)
        if kp is not None:
            all_kp.append(kp)
    cap.release()

    if len(all_kp) <= max_raw_frames:
        return all_kp

    idx = np.linspace(0, len(all_kp) - 1, max_raw_frames).round().astype(int)
    return [all_kp[i] for i in idx]


# =============================================
# Noi suy (da fix cubic-interpolation crash)
# =============================================

def interpolate_keypoints(seq: list, target_len: int):
    
    if not seq:
        return None

    arr = np.stack(seq, axis=0).astype(np.float32)   # (T, F)
    T, F = arr.shape

    if T == target_len:
        return arr
    if T == 1:
        return np.repeat(arr, target_len, axis=0)

    kind = 'cubic' if T >= 4 else 'linear'

    t_orig   = np.linspace(0, 1, T)
    t_target = np.linspace(0, 1, target_len)

    result = np.zeros((target_len, F), dtype=np.float32)
    for fi in range(F):
        f = interp1d(t_orig, arr[:, fi], kind=kind,
                     bounds_error=False, fill_value='extrapolate')
        result[:, fi] = f(t_target)

    return result


# =============================================
# Tao mau tang cuong
# =============================================

def generate_augmented_samples(
    original_seq: list,
    aug_funcs: list,
    num_samples: int,
    max_augs: int,
) -> list:
   
    results = []
    n_funcs = len(aug_funcs)
    if not original_seq or n_funcs == 0:
        return results

    for _ in range(num_samples):
        seq = [kp.copy() for kp in original_seq]
        n_apply = random.randint(1, min(max_augs, n_funcs))
        chosen  = random.sample(aug_funcs, n_apply)
        random.shuffle(chosen)

        for fn in chosen:
            seq = fn(seq)
            if not seq:
                break

        if seq:
            results.append(seq)

    return results


# =============================================
# Xu ly mot video (chay trong worker process)
# =============================================

def _init_worker():
   
    random.seed()


def process_one_video(args, cfg: Config):
    
    video_path, action_path, action, label, num_aug, start_idx = args

    
    video_basename = os.path.basename(video_path)
    expected_files = [
        os.path.join(action_path, f"{i}.npz")
        for i in range(start_idx, start_idx + num_aug + 1)
    ]
    if cfg.resume and all(os.path.exists(f) for f in expected_files):
        rows = [
            (f, label, action, video_basename)
            for f in expected_files
        ]
        return action, rows, len(rows), None

    try:
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=cfg.model_complexity,
        ) as holistic:
            frame_list = read_video_keypoints(video_path, holistic, cfg.max_raw_frames)

        if not frame_list:
            return action, [], 0, f"Khong trich xuat duoc keypoint: {video_path}"

        augmented = generate_augmented_samples(
            frame_list, AUGMENTATIONS, num_aug, cfg.max_augs_per_sample
        )
        all_seqs = augmented + [frame_list]

       

        idx = start_idx
        rows = []
        for seq in all_seqs:
            interp = interpolate_keypoints(seq, cfg.sequence_length)
            if interp is None:
                continue
            file_path = os.path.join(action_path, f"{idx}.npz")
            np.savez_compressed(
                file_path, sequence=interp, label=label, video=video_basename
            )
            rows.append((file_path, label, action, video_basename))
            idx += 1

        return action, rows, len(rows), None

    except Exception as e:
        return action, [], 0, f"Loi xu ly {video_path}: {type(e).__name__}: {e}"


# =============================================
# Manifest
# =============================================

def write_manifest(log_path: str, rows: list, label_map: dict):
    manifest_path = os.path.join(log_path, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "label", "action", "video"])
        for file_path, label, action, video in rows:
            writer.writerow([file_path, label, action, video])
    return manifest_path


# =============================================
# ARGPARSE
# =============================================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Trich xuat keypoint MediaPipe + tao du lieu augment cho VSL")

    p.add_argument("--dataset-path", default="Dataset")
    p.add_argument("--data-path", default="Data", help="Thu muc output chua .npz theo tung action")
    p.add_argument("--log-path", default="Logs")
    p.add_argument("--sequence-length", type=int, default=60)
    p.add_argument("--max-raw-frames", type=int, default=80)
    p.add_argument("--num-aug-samples", type=int, default=8)
    p.add_argument("--max-augs-per-sample", type=int, default=3)
    p.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2],
                    help="Do phuc tap MediaPipe Holistic: 0 nhanh nhat, 2 chinh xac nhat")
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    p.add_argument("--resume", action="store_true", help="Bo qua video da xu ly xong tu lan chay truoc")
    p.add_argument("--limit", type=int, default=None, help="Chi xu ly N video dau tien (de test)")

    a = p.parse_args()

    return Config(
        dataset_path=a.dataset_path,
        video_folder=os.path.join(a.dataset_path, "Videos"),
        label_file=os.path.join(a.dataset_path, "Text", "label.csv"),
        data_path=a.data_path,
        log_path=a.log_path,
        sequence_length=a.sequence_length,
        max_raw_frames=a.max_raw_frames,
        num_aug_samples=a.num_aug_samples,
        max_augs_per_sample=a.max_augs_per_sample,
        model_complexity=a.model_complexity,
        workers=a.workers,
        resume=a.resume,
        limit=a.limit,
    )


# =============================================
# MAIN
# =============================================

def main():
    cfg = parse_args()
    os.makedirs(cfg.data_path, exist_ok=True)
    os.makedirs(cfg.log_path, exist_ok=True)
    logger = setup_logger(cfg.log_path)

    logger.info("=" * 60)
    logger.info("CREATE KEYPOINT DATASET (MediaPipe) FOR BiLSTM")
    logger.info("=" * 60)

    if not os.path.exists(cfg.label_file):
        logger.error(f"Khong tim thay file nhan: {cfg.label_file}")
        return

    df = pd.read_csv(cfg.label_file)

    required_cols = {"LABEL", "VIDEO"}
    if not required_cols.issubset(df.columns):
        logger.error(f"File nhan thieu cot bat buoc {required_cols}, hien co: {list(df.columns)}")
        return

    selected_actions = sorted(df["LABEL"].unique())
    label_map = {action: idx for idx, action in enumerate(selected_actions)}

    label_map_path = os.path.join(cfg.log_path, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    logger.info(f"Tong so nhan: {len(selected_actions)}")
    logger.info(f"Tong so video trong CSV: {len(df)}")
    logger.info(f"N_FEATURES = {N_FEATURES} (phai khop N_FEATURES trong notebook train)")
    logger.info(f"Moi video tao {cfg.num_aug_samples} mau aug + 1 goc")
    logger.info(f"Model complexity MediaPipe: {cfg.model_complexity}")
    logger.info(f"Resume mode: {cfg.resume}")
    logger.info(f"label_map.json: {label_map_path}")

    for action in selected_actions:
        os.makedirs(os.path.join(cfg.data_path, action), exist_ok=True)

    action_idx_counter = {a: 0 for a in selected_actions}
    tasks = []

    for _, row in df.iterrows():
        action = row["LABEL"]
        video_file = row["VIDEO"]
        label = label_map[action]
        video_path = os.path.join(cfg.video_folder, video_file)

        if not os.path.exists(video_path):
            logger.warning(f"Khong tim thay video: {video_path}")
            continue

        action_path = os.path.join(cfg.data_path, action)
        start_idx = action_idx_counter[action]
        action_idx_counter[action] += cfg.num_aug_samples + 1

        tasks.append((video_path, action_path, action, label, cfg.num_aug_samples, start_idx))

    if cfg.limit:
        tasks = tasks[:cfg.limit]

    logger.info(f"So video se xu ly: {len(tasks)}")
    logger.info(f"CPU workers: {cfg.workers}")

    timer = Timer()
    total_saved = 0
    all_rows = []
    errors = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cfg.workers, initializer=_init_worker
    ) as executor:
        futures = {executor.submit(process_one_video, t, cfg): t for t in tasks}
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Xu ly video",
        ):
            action_name, rows, n_saved, err = fut.result()
            total_saved += n_saved
            all_rows.extend(rows)
            if err:
                errors.append(err)
                logger.warning(err)
            else:
                logger.info(f"{action_name}: {n_saved} mau - {timer.elapsed()}")

    manifest_path = write_manifest(cfg.log_path, all_rows, label_map)

    logger.info("")
    logger.info("-" * 50)
    logger.info(f"HOAN THANH. Tong mau tao: {total_saved}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Loi: {len(errors)}")
    if len(errors) > 20:
        logger.warning(f"... va {len(errors) - 20} loi khac, xem day du trong log file")
    logger.info(f"Tong thoi gian: {timer.elapsed()}")
    logger.info("-" * 50)


if __name__ == "__main__":
    main()
