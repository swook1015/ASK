import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from collections import deque
from ultralytics import YOLO
import re

# 💡 텐서플로우 하드웨어 가속 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

# ==========================================
# ⚙️ 1. 설정 및 경로
# ==========================================
VIDEO_DIR = './AI/dataset/nturgb+d_rgb_C001_fall'
YOLO_MODEL_PATH = './AI/models/yolo/yolo8n-pose_full_integer_quant.tflite'
LN_MODEL_PATH   = './AI/models/ln/new_lpn_cv_60_legacy.tflite'
TCN_MODEL_PATH  = './AI/models/1d-tcn/tcn_manual_fixed.tflite'

WINDOW_SIZE = 60
FALL_THRESHOLD = 0.5

# ==========================================
# 🧠 2. 모델 로드 및 파이프라인 함수
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

def center_poses_2d_final(X_2d_window):
    # 0번(골반중심) 기준 영점 조절
    root = X_2d_window[:, 0:1, :] 
    centered = X_2d_window - root
    # 💡 [필수] Y축 반전 (NTU 좌표계 동기화)
    centered[:, :, 1] = -centered[:, :, 1] 
    max_dist = np.max(np.abs(centered)) + 1e-6
    return (centered / max_dist).reshape(1, WINDOW_SIZE, 26).astype(np.float32)

# ==========================================
# 🎥 3. 메인 루프 (영상 강제 확장 로직)
# ==========================================
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
tp, fn = 0, 0 # A043 전용 Recall 계산용

for v_path in video_files:
    filename = os.path.basename(v_path)
    
    # 정답 추출 (A043~A049: FALL)
    action_match = re.search(r'A(\d{3})', filename)
    action_id = int(action_match.group(1)) if action_match else 0
    is_actual_fall = 43 <= action_id <= 49

    cap = cv2.VideoCapture(v_path)
    raw_kpts_list = []

    # [STEP 1] 영상의 모든 관절을 먼저 리스트에 담기 (AI 안 꺼지게)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = yolo_model.predict(frame, verbose=False, conf=0.15)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            curr_kpts = results[0].keypoints.xy[0].cpu().numpy()
            hip_center = (curr_kpts[11] + curr_kpts[12]) / 2.0
            kpts_13 = np.vstack((hip_center, curr_kpts[5:17])) # 13관절 매핑
            raw_kpts_list.append(kpts_13)
        else:
            # 관절 놓치면 이전 프레임 복사해서 땜빵
            raw_kpts_list.append(raw_kpts_list[-1] if raw_kpts_list else np.zeros((13, 2)))
    cap.release()

    if not raw_kpts_list: continue

    # [STEP 2] 💡 [핵심] 영상 강제 확장 (Stretching)
    # 1프레임을 앞에 60개 깔고, 마지막 프레임을 뒤에 60개 깔아서 AI가 끝까지 보게 함
    padded_sequence = [raw_kpts_list[0]] * (WINDOW_SIZE - 1) + raw_kpts_list + [raw_kpts_list[-1]] * (WINDOW_SIZE - 1)
    
    max_score = 0.0
    kpts_buffer = deque(maxlen=WINDOW_SIZE)

    # [STEP 3] 확장된 시퀀스로 추론 진행
    for kpts in padded_sequence:
        kpts_buffer.append(kpts)
        if len(kpts_buffer) == WINDOW_SIZE:
            input_2d = center_poses_2d_final(np.array(kpts_buffer))
            
            # 양자화 파이프라인
            input_2d_q = np.clip(np.round(input_2d / ln_in_s + ln_in_z), -128, 127).astype(np.int8)
            ln_int.set_tensor(ln_in['index'], input_2d_q)
            ln_int.invoke()
            ln_out_f = (ln_int.get_tensor(ln_out['index']).astype(np.float32) - ln_out_z) * ln_out_s
            
            tcn_in_q = np.clip(np.round(ln_out_f / tcn_in_s + tcn_in_z), -128, 127).astype(np.int8)
            tcn_int.set_tensor(tcn_in['index'], tcn_in_q)
            tcn_int.invoke()
            res_f = (tcn_int.get_tensor(tcn_out['index']).astype(np.float32) - tcn_out_z) * tcn_out_s
            
            score = res_f.flatten()[0]
            if score > max_score: max_score = score

    # [STEP 4] 결과 집계
    prediction = "FALL" if max_score > FALL_THRESHOLD else "NORMAL"
    is_correct = (prediction == "FALL" if is_actual_fall else prediction == "NORMAL")
    
    if is_actual_fall:
        if is_correct: tp += 1
        else: fn += 1

    print(f"▶ [{filename}] MaxScore: {max_score:.4f} | 판정: {prediction} | {'✅ 정답' if is_correct else '❌ 오답'}")

# ==========================================
# 📊 4. 최종 Recall 계산
# ==========================================
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print("\n" + "="*50)
print(f"최종 리포트 (A043~A049 기준)")
print(f"Recall (재현율): {recall:.4f} ({tp}/{tp+fn})")
print(f"Accuracy (정확도): {recall:.4f}") # 현재 전부 낙상 영상이므로 동일
print("="*50)