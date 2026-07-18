import cv2
import mediapipe as mp
import time
import os
import glob
import re

# =====================================================
# CONFIG
# =====================================================
CAMERA_INDEX     = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
TARGET_FPS       = 30
OUTPUT_FOLDER    = "Dataset1/tuoi"
SHOW_PREVIEW     = True
COUNTDOWN        = 3
READY_THRESHOLD  = 5    
MAX_RECORD_FRAMES = 120  
MIN_RECORD_FRAMES = 15  
                         

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils


def next_video_index():
    files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.mp4"))
    maximum = 0
    for f in files:
        m = re.search(r"(\d+)\.mp4$", f)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return maximum + 1


# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


REAL_WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_WIDTH
REAL_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
REAL_FPS_CAM = cap.get(cv2.CAP_PROP_FPS)
if not REAL_FPS_CAM or REAL_FPS_CAM <= 1:
    REAL_FPS_CAM = TARGET_FPS
print(f"[INFO] Camera thuc te: {REAL_WIDTH}x{REAL_HEIGHT} @ {REAL_FPS_CAM:.1f}fps")

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# =====================================================
# HOLISTIC
# =====================================================
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# =====================================================
# FPS DO THUC TE (de hien thi HUD)
# =====================================================
last_time = time.time()
fps_counter = 0
real_fps = 0.0


def update_fps():
    global last_time, fps_counter, real_fps
    fps_counter += 1
    now = time.time()
    if now - last_time >= 1:
        real_fps = fps_counter / (now - last_time)
        fps_counter = 0
        last_time = now


# =====================================================
# HELPERS
# =====================================================

def is_ready(results):
    """[FIX] Doi ten thanh is_ready() cho khop voi noi goi ben duoi."""
    if results.pose_landmarks is None:
        return False
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        return False
    return True


def draw_landmarks(frame, results):
    """[FIX] Ham nay bi thieu hoan toan trong ban goc."""
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1),
        )
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(121, 44, 250), thickness=1),
        )
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=2),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(245, 66, 230), thickness=1),
        )


def draw_hud(frame, is_recording, ready_state, video_index):
    color = (0, 255, 0)
    if not ready_state:
        color = (0, 0, 255)
    status = "READY" if ready_state else "NOT READY"
    if is_recording:
        status = "RECORDING"

    cv2.rectangle(frame, (10, 10), (280, 130), (40, 40, 40), -1)
    cv2.putText(frame, f"Status : {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS : {real_fps:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Video : {video_index}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def countdown():
    """[FIX] Them cv2.imshow() de nguoi dung THAY duoc so dem nguoc."""
    for i in range(COUNTDOWN, 0, -1):
        t0 = time.time()
        while time.time() - t0 < 1:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame, str(i), (280, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8,
            )
            cv2.imshow("Recorder", frame)   # [FIX] dong nay bi thieu o ban goc
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
    return True


def record_one_video(video_index):
    
    ready_counter = 0          # [FIX] thieu khoi tao o ban goc
    stable_ready = False

    # ---------------- PREVIEW ----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        update_fps()

        ready_now = is_ready(results)          # [FIX] goi dung ten ham
        if ready_now:
            ready_counter += 1
        else:
            ready_counter = 0
        stable_ready = ready_counter >= READY_THRESHOLD

        draw_landmarks(frame, results)
        draw_hud(frame, False, stable_ready, video_index)
        cv2.putText(frame, "SPACE : Record   |   Q : Quit", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"                        # [FIX] thoat han, KHONG ghi file
        if key == 32 and stable_ready:
            if not countdown():
                return "quit"                     # nguoi dung bam q trong luc dem nguoc
            break

    frame_buffer = []          
    frame_counter = 0
    start_time = time.time()
    recording = True

    while recording:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        update_fps()

        frame_buffer.append(frame.copy())  
        frame_counter += 1

        draw_landmarks(frame, results)
        draw_hud(frame, True, True, video_index)

        duration_live = time.time() - start_time
        cv2.putText(frame, f"Time : {duration_live:.1f}s", (330, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frames : {frame_counter}", (330, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bar_width = 220
        progress = min(frame_counter / MAX_RECORD_FRAMES, 1.0)
        cv2.rectangle(frame, (330, 90), (330 + bar_width, 110), (100, 100, 100), 2)
        cv2.rectangle(frame, (330, 90),
                      (330 + int(progress * bar_width), 110), (0, 255, 0), -1)

        cv2.putText(frame, "SPACE : Stop Recording", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 32:
            recording = False

    total_duration = time.time() - start_time   # [FIX] thoi luong quay THUC TE

    if frame_counter < MIN_RECORD_FRAMES:
        print(f"[WARN] Video qua ngan ({frame_counter} frames < {MIN_RECORD_FRAMES}), "
              f"da xoa. Vui long ghi lai.")
        return "retry"                           

   
    actual_fps = frame_counter / total_duration if total_duration > 0 else REAL_FPS_CAM
   
    actual_fps = max(5.0, min(actual_fps, REAL_FPS_CAM))

    output_name = os.path.join(OUTPUT_FOLDER, f"{video_index}.mp4")
    writer = cv2.VideoWriter(
        output_name, FOURCC, actual_fps, (REAL_WIDTH, REAL_HEIGHT)   # [FIX] dung actual_fps
    )
    if not writer.isOpened():
        print("Cannot create video.")
        return "quit"

    for f in frame_buffer:            # [FIX] ghi toan bo frame da buffer, sau khi biet fps dung
        writer.write(f)
    writer.release()

    # [FIX] Da bo phan ghi file .json metadata theo yeu cau.
    print(f"Saved {output_name}  ({frame_counter} frames, {round(actual_fps, 2)}fps, {round(total_duration, 2)}s)")
    return "next"


# =====================================================
# MAIN LOOP  [FIX] cho phep quay lien tiep nhieu video
# =====================================================
try:
    video_index = next_video_index()
    while True:
        status = record_one_video(video_index)
        if status == "quit":
            break
        elif status == "retry":
            
            continue
        else:  # "next"
            video_index += 1
finally:
    
    holistic.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")