import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from collections import deque
from ultralytics import YOLO
import re

# 💡 하드웨어 가속 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

# ==========================================
# ⚙️ 1. 설정 및 경로
# ==========================================
VIDEO_DIR = './AI/dataset/nturgb+d_rgb_C001_fall'
OUTPUT_DIR = './output_debug'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_MODEL_PATH = './AI/models/yolo/yolo8n-pose_full_integer_quant.tflite'
LN_MODEL_PATH   = './AI/models/ln/0408new_lpn_cv_60_legacy.tflite'
TCN_MODEL_PATH  = './AI/models/1d-tcn/0408tcn_8layer_pure_ntu.tflite'

WINDOW_SIZE = 60
# 🚨 아까 분석했던 최적의 기준점 0.9 적용! (오탐지 완벽 방어)
FALL_THRESHOLD = 0.9 

# 🦴 스켈레톤 연결 정보
SKELETON_MAP = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (0, 7), (0, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)
]

# ==========================================
# 🧠 2. 모델 로드 및 파이프라인
# ==========================================
yolo_model = YOLO(YOLO_MODEL_PATH)
def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]

ln_int, ln_in, ln_out = load_tflite(LN_MODEL_PATH)
tcn_int, tcn_in, tcn_out = load_tflite(TCN_MODEL_PATH)

ln_in_s, ln_in_z = ln_in['quantization']
ln_out_s, ln_out_z = ln_out['quantization']
tcn_in_s, tcn_in_z = tcn_in['quantization']
tcn_out_s, tcn_out_z = tcn_out['quantization']

def center_poses_2d_final(X_2d_window, global_max_dist):
    root = X_2d_window[:, 0:1, :] 
    centered = X_2d_window - root
    centered[:, :, 1] = -centered[:, :, 1] 
    return (centered / global_max_dist).reshape(1, WINDOW_SIZE, 26).astype(np.float32)

# ==========================================
# 🎥 3. 메인 루프 (시각화 저장 포함)
# ==========================================
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))

# 🚨 수정: 정확도 계산을 위해 모든 경우의 수(TP, FN, TN, FP) 추적
tp, fn, tn, fp = 0, 0, 0, 0 

for v_path in video_files:
    filename = os.path.basename(v_path)
    action_match = re.search(r'A(\d{3})', filename)
    action_id = int(action_match.group(1)) if action_match else 0
    
    is_actual_fall = (action_id == 43)

    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    
    raw_frames = []
    raw_kpts_list = []

    # [STEP 1] 원본 영상 로드 및 관절 추출
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        raw_frames.append(frame)
        
        results = yolo_model.predict(frame, verbose=False, conf=0.15, classes=[0])
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            curr_kpts = results[0].keypoints.xy[0].cpu().numpy()
            hip_center = (curr_kpts[11] + curr_kpts[12]) / 2.0
            kpts_13 = np.vstack((hip_center, curr_kpts[5:17])) 
            raw_kpts_list.append(kpts_13)
        else:
            raw_kpts_list.append(raw_kpts_list[-1] if raw_kpts_list else np.zeros((13, 2)))
    cap.release()

    if not raw_kpts_list: continue

    # [영상 전체 기준 스케일 고정]
    temp_kpts = np.array(raw_kpts_list)
    temp_root = temp_kpts[:, 0:1, :]
    temp_centered = temp_kpts - temp_root
    global_max_dist = np.max(np.abs(temp_centered)) + 1e-6

    # [STEP 2] 영상 및 시퀀스 강제 확장
    padded_kpts = [raw_kpts_list[0]] * (WINDOW_SIZE - 1) + raw_kpts_list + [raw_kpts_list[-1]] * WINDOW_SIZE
    padded_frames = [raw_frames[0]] * (WINDOW_SIZE - 1) + raw_frames + [raw_frames[-1]] * WINDOW_SIZE
    
    out_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    max_score = 0.0
    kpts_buffer = deque(maxlen=WINDOW_SIZE)

    # [STEP 3] 확장된 시퀀스로 추론 및 시각화 저장
    for idx, (kpts, frame) in enumerate(zip(padded_kpts, padded_frames)):
        kpts_buffer.append(kpts)
        current_frame_score = 0.0
        
        if len(kpts_buffer) == WINDOW_SIZE:
            input_2d = center_poses_2d_final(np.array(kpts_buffer), global_max_dist)
            
            # 양자화 추론
            input_2d_q = np.clip(np.round(input_2d / ln_in_s + ln_in_z), -128, 127).astype(np.int8)
            ln_int.set_tensor(ln_in['index'], input_2d_q)
            ln_int.invoke()
            ln_out_f = (ln_int.get_tensor(ln_out['index']).astype(np.float32) - ln_out_z) * ln_out_s
            
            tcn_in_q = np.clip(np.round(ln_out_f / tcn_in_s + tcn_in_z), -128, 127).astype(np.int8)
            tcn_int.set_tensor(tcn_in['index'], tcn_in_q)
            tcn_int.invoke()
            res_f = (tcn_int.get_tensor(tcn_out['index']).astype(np.float32) - tcn_out_z) * tcn_out_s
            
            current_frame_score = res_f.flatten()[0]
            if current_frame_score > max_score: max_score = current_frame_score

        # 시각화 
        ann_frame = frame.copy()
        for connection in SKELETON_MAP:
            pt1 = tuple(kpts[connection[0]].astype(int))
            pt2 = tuple(kpts[connection[1]].astype(int))
            cv2.line(ann_frame, pt1, pt2, (0, 255, 255), 2)
        for kp in kpts:
            cv2.circle(ann_frame, tuple(kp.astype(int)), 4, (255, 0, 0), -1)

        status = "FALL!!" if current_frame_score > FALL_THRESHOLD else "NORMAL"
        color = (0, 0, 255) if status == "FALL!!" else (0, 255, 0)
        cv2.putText(ann_frame, f"{status} ({current_frame_score:.2f})", (30, 60), 2, 1.2, color, 3)
        out_writer.write(ann_frame)

    out_writer.release()

    # 🚨 [STEP 4] 결과 집계 로직 전면 수정 (모든 수치 추적)
    prediction = "FALL" if max_score > FALL_THRESHOLD else "NORMAL"
    is_correct = (prediction == "FALL" if is_actual_fall else prediction == "NORMAL")
    
    if is_actual_fall:
        if prediction == "FALL": tp += 1
        else: fn += 1
    else:
        if prediction == "NORMAL": tn += 1
        else: fp += 1

    print(f"▶ [{filename}] MaxScore: {max_score:.4f} | 판정: {prediction} | {'✅ 정답' if is_correct else '❌ 오답'}")

# ==========================================
# 📊 5. 최종 리포트 출력
# ==========================================
total_videos = tp + fn + tn + fp
total_falls = tp + fn
total_normals = tn + fp

print("\n" + "="*50)
print("📊 [End-to-End 최종 성능 리포트]")
print("="*50)

if total_videos > 0:
    accuracy = (tp + tn) / total_videos
    recall = tp / total_falls if total_falls > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"✅ Accuracy (정확도) : {accuracy*100:.2f}% ({(tp+tn)}/{total_videos})")
    print(f"✅ Precision(정밀도) : {precision*100:.2f}%")
    print(f"✅ Recall   (재현율) : {recall*100:.2f}% ({tp}/{total_falls})")
    print(f"✅ F1-Score          : {f1_score*100:.2f}%")
    print("-" * 50)
    print("📋 [Confusion Matrix]")
    print(f"  TN (정상->정상) : {tn} | FP (정상->낙상) : {fp}")
    print(f"  FN (낙상->정상) : {fn}  | TP (낙상->낙상) : {tp}")
else:
    print("처리된 영상이 없습니다.")
print("="*50)