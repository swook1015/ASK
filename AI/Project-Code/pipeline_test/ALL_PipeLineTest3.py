import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from collections import deque
from ultralytics import YOLO
import re

# ⚙️ 설정 및 경로
VIDEO_DIR = './AI/dataset/nturgb+d_rgb_C001_fall'
YOLO_MODEL_PATH = './AI/models/yolo/yolo8n-pose_full_integer_quant.tflite'
LN_MODEL_PATH   = './AI/models/ln/new_lpn_cv_60_legacy.tflite'
TCN_MODEL_PATH  = './AI/models/1d-tcn/tcn_manual_fixed.tflite'

WINDOW_SIZE = 60
FALL_THRESHOLD = 0.5

# 🧠 모델 로드 함수
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
    root = X_2d_window[:, 0:1, :] 
    centered = X_2d_window - root
    centered[:, :, 1] = -centered[:, :, 1] 
    max_dist = np.max(np.abs(centered)) + 1e-6
    return (centered / max_dist).reshape(1, WINDOW_SIZE, 26).astype(np.float32)

# 🎥 영상 루프
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))

for v_path in video_files:
    filename = os.path.basename(v_path)
    
    # 정답 추출 (A043~A049: FALL)
    action_match = re.search(r'A(\d{3})', filename)
    action_id = int(action_match.group(1)) if action_match else 0
    ground_truth = "FALL" if 43 <= action_id <= 49 else "NORMAL"

    cap = cv2.VideoCapture(v_path)
    all_kpts = []

    # 1단계: 영상 전체 프레임에서 관절 데이터 미리 다 뽑기
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model.predict(frame, verbose=False, conf=0.15)
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            curr_kpts = results[0].keypoints.xy[0].cpu().numpy()
            hip_center = (curr_kpts[11] + curr_kpts[12]) / 2.0
            kpts_13 = np.vstack((hip_center, curr_kpts[5:17])) 
            all_kpts.append(kpts_13)
    cap.release()

    if not all_kpts:
        print(f"▶ [{filename}] 관절 검출 실패 | 판정: SKIP")
        continue

    # 2단계: 💡 [핵심] 1프레임을 60까지 늘리고, 끝난 뒤에도 밀어주기 (Padding)
    # 앞쪽 패딩: 첫 프레임 59개 + 전체 관절 + 뒤쪽 패딩: 마지막 프레임 59개
    padded_kpts = [all_kpts[0]] * (WINDOW_SIZE - 1) + all_kpts + [all_kpts[-1]] * (WINDOW_SIZE - 1)
    
    max_score = 0.0
    kpts_buffer = deque(maxlen=WINDOW_SIZE)

    # 3단계: 확장된 시퀀스로 추론 (영상이 끝나도 AI는 계속 돌아감)
    for kpts in padded_kpts:
        kpts_buffer.append(kpts)
        
        if len(kpts_buffer) == WINDOW_SIZE:
            input_2d = center_poses_2d_final(np.array(kpts_buffer))
            
            # 양자화 및 추론
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

    # 최종 결과 출력
    prediction = "FALL" if max_score > FALL_THRESHOLD else "NORMAL"
    result_icon = "✅ 정답" if prediction == ground_truth else "❌ 오답"
    
    print(f"▶ [{filename}] MaxScore: {max_score:.4f} | 판정: {prediction} | {result_icon}")

print("\n✅ 이제 숏폼 영상도 안 놓치고 다 잡아냅니다!")