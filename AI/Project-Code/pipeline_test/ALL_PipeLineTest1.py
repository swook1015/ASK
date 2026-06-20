import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from collections import deque
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, recall_score

# 💡 [멈춤 방지] 텐서플로우 엔진 초기화 락 방지 환경 변수
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

# ==========================================
# ⚙️ 1. 경로 및 설정
# ==========================================
VIDEO_DIR = './AI/dataset/nturgb+d_rgb_C001_fall' 

YOLO_MODEL_PATH = './AI/models/yolo/yolo26n-pose_full_integer_quant.tflite'
LN_MODEL_PATH   = './AI/models/ln/new_lpn_cv_60_legacy.tflite'
TCN_MODEL_PATH  = './AI/models/1d-tcn/tcn_manual_fixed.tflite'

WINDOW_SIZE = 60
FALL_THRESHOLD = 0.5 

# 리사이즈 규격 고정 (CPU 부하 감소)
RESIZE_W, RESIZE_H = 640, 360

# ==========================================
# 🧠 2. 모델 로드 및 엔진 예열
# ==========================================
print("🚀 모델 로드 및 엔진 예열 중...")
yolo_model = YOLO(YOLO_MODEL_PATH)
# 초기 로딩 지연 방지를 위한 더미 추론
yolo_model.predict(np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8), verbose=False)

def load_tflite_fast(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]

ln_int, ln_in, ln_out = load_tflite_fast(LN_MODEL_PATH)
tcn_int, tcn_in, tcn_out = load_tflite_fast(TCN_MODEL_PATH)

# 양자화 파라미터 추출
ln_in_s, ln_in_z = ln_in['quantization']
ln_out_s, ln_out_z = ln_out['quantization']
tcn_in_s, tcn_in_z = tcn_in['quantization']
tcn_out_s, tcn_out_z = tcn_out['quantization']

# 💡 [교정 1] 전처리 함수: Y축 반전 및 2D Max Dist 정규화
def center_poses_2d_final(X_2d_window):
    # 1. 골반(0번) 기준 영점 조절
    root = X_2d_window[:, 0:1, :]
    centered = X_2d_window - root
    
    # 2. 🔥 Y축 반전 (이미지 좌표계 Down -> NTU 3D 좌표계 Up)
    centered[:, :, 1] = -centered[:, :, 1]
    
    # 3. Max Dist 정규화 (-1 ~ 1)
    max_dist = np.max(np.abs(centered)) + 1e-6
    normalized = centered / max_dist
    
    return normalized.reshape(1, WINDOW_SIZE, 26).astype(np.float32)

# ==========================================
# 🎥 3. 영상 루프
# ==========================================
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
y_true_list, y_pred_list = [], []

print(f"🔍 총 {len(video_files)}개 영상 분석 시작 (일관성 교정 모드)\n")

for v_path in video_files:
    filename = os.path.basename(v_path)
    action_id = int(filename.split('A')[1][:3])
    label = 1 if action_id == 43 else 0 
    
    cap = cv2.VideoCapture(v_path)
    kpts_buffer = deque(maxlen=WINDOW_SIZE)
    video_scores = [] 
    is_first = True 

    print(f"▶ [{filename}] 분석...", end="", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO 추론 (리사이즈 적용)
        frame_small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        results = yolo_model.predict(frame_small, verbose=False, conf=0.3)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            all_kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # 좌표 정규화 (0~1)
            all_kpts[:, 0] /= RESIZE_W
            all_kpts[:, 1] /= RESIZE_H
            
            # 골반 중심 및 13관절 매핑
            hip_center = (all_kpts[11] + all_kpts[12]) / 2.0
            kpts_13 = np.vstack((hip_center, all_kpts[5:11], all_kpts[11:17]))
            
            # 버퍼 채우기
            if is_first:
                for _ in range(WINDOW_SIZE): kpts_buffer.append(kpts_13)
                is_first = False
            else:
                kpts_buffer.append(kpts_13)
            
        # 2. 추론 파이프라인
        if len(kpts_buffer) == WINDOW_SIZE:
            window_data = np.array(kpts_buffer)
            
            # [LPN 전처리] Y축 반전 포함
            input_2d = center_poses_2d_final(window_data)
            
            # [LPN 추론]
            input_2d_q = np.clip(np.round(input_2d / ln_in_s + ln_in_z), -128, 127).astype(np.int8)
            ln_int.set_tensor(ln_in['index'], input_2d_q)
            ln_int.invoke()
            ln_out_f = (ln_int.get_tensor(ln_out['index']).astype(np.float32) - ln_out_z) * ln_out_s
            
            # 💡 [교정 2] 3D Max Dist 정규화 (TCN 학습 환경과 통일)
            max_dist_3d = np.max(np.abs(ln_out_f)) + 1e-6
            ln_out_norm = ln_out_f / max_dist_3d
            
            # [TCN 추론] ln_out_norm(정규화된 3D) 투입
            tcn_in_q = np.clip(np.round(ln_out_norm / tcn_in_s + tcn_in_z), -128, 127).astype(np.int8)
            tcn_int.set_tensor(tcn_in['index'], tcn_in_q)
            tcn_int.invoke()
            res_f = (tcn_int.get_tensor(tcn_out['index']).astype(np.float32) - tcn_out_z) * tcn_out_s
            
            # 최종 스코어 (Sigmoid 출력값)
            score = res_f.flatten()[0]
            video_scores.append(score)

    cap.release()

    if video_scores:
        video_scores.sort(reverse=True)
        # 상위 5개 프레임 평균 점수로 최종 판정
        max_score = np.mean(video_scores[:5]) 
        final_pred = 1 if max_score > FALL_THRESHOLD else 0
        
        y_true_list.append(label)
        y_pred_list.append(final_pred)
        
        status = "✅ 정답" if label == final_pred else "❌ 오답"
        print(f"\r▶ [{filename}] MaxScore: {max_score:.4f} | 판정: {'FALL' if final_pred else 'NORMAL'} | {status}")

# ==========================================
# 📊 4. 최종 리포트
# ==========================================
print("\n" + "="*60)
if len(y_true_list) > 0:
    acc = accuracy_score(y_true_list, y_pred_list)
    rec = recall_score(y_true_list, y_pred_list, zero_division=0)
    print(f"📊 파이프라인 일관성 교정 결과")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - Recall:   {rec*100:.2f}%")
print("="*60)