import os
import cv2
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
OUTPUT_DIR = './output' 

# 출력 폴더 자동 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_MODEL_PATH = './AI/models/yolo/yolo8n-pose_full_integer_quant.tflite'
LPN_MODEL_PATH  = './AI/models/ln/new_lpn_cv_60_legacy.tflite'
TCN_MODEL_PATH  = './AI/models/1d-tcn/tcn_manual_fixed.tflite'

WINDOW_SIZE = 60
FALL_THRESHOLD = 0.5 
RESIZE_W, RESIZE_H = 640, 360 

YOLO_TO_LPN = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ==========================================
# 🧠 2. AI 모델 로드 
# ==========================================
print("🚀 모델 로드 및 엔진 예열 중...")
yolo_model = YOLO(YOLO_MODEL_PATH, task='pose')
yolo_model.predict(np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8), verbose=False)

def load_tflite_fast(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]

ln_int, ln_in, ln_out = load_tflite_fast(LPN_MODEL_PATH)
tcn_int, tcn_in, tcn_out = load_tflite_fast(TCN_MODEL_PATH)

ln_in_s, ln_in_z = ln_in['quantization']
ln_out_s, ln_out_z = ln_out['quantization']
tcn_in_s, tcn_in_z = tcn_in['quantization']
tcn_out_s, tcn_out_z = tcn_out['quantization']

# ==========================================
# 🎥 3. 영상 루프 (관절 추출 + 관성 예측 + 영상 렌더링)
# ==========================================
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
y_true_list, y_pred_list = [], []

print(f"🔍 총 {len(video_files)}개 영상 분석 및 렌더링 시작\n")

for v_path in video_files:
    filename = os.path.basename(v_path)
    action_id = int(filename.split('A')[1][:3])
    label = 1 if action_id == 43 else 0 
    
    cap = cv2.VideoCapture(v_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30
    out_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (RESIZE_W, RESIZE_H))
    
    joint_buffer = []
    video_scores = [] 
    
    # 실시간 관성 예측용 변수
    prev_kps_pixel = None 
    prev_velocity = np.zeros((13, 2)) 
    
    current_score = 0.0 

    print(f"▶ [{filename}] 분석 및 렌더링 중...", end="", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        
        results = yolo_model.predict(frame_small, verbose=False, conf=0.15)
        annotated_frame = results[0].plot(labels=False, boxes=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            
            # 면적 기준 1등 타겟팅
            boxes = results[0].boxes.xywh.cpu().numpy()
            areas = boxes[:, 2] * boxes[:, 3]
            best_idx = np.argmax(areas)
            
            curr_kpts_norm = results[0].keypoints.xyn[best_idx].cpu().numpy()
            target_kps = curr_kpts_norm[YOLO_TO_LPN] 
            curr_pixel = target_kps * 416.0 
            
            # 속도 기반 관성 예측 보간 로직
            if prev_kps_pixel is not None:
                zero_mask = (curr_kpts_norm[YOLO_TO_LPN, 0] == 0) & (curr_kpts_norm[YOLO_TO_LPN, 1] == 0)
                valid_mask = ~zero_mask
                
                # 살아있는 관절은 하강 속도 갱신
                prev_velocity[valid_mask] = curr_pixel[valid_mask] - prev_kps_pixel[valid_mask]
                
                # 죽은 관절(0,0)은 이전 위치에 하강 속도(관성)를 더해서 부드럽게 떨어지게 함
                curr_pixel[zero_mask] = prev_kps_pixel[zero_mask] + (prev_velocity[zero_mask] * 0.8)
            
            if prev_kps_pixel is None:
                prev_kps_pixel = curr_pixel.copy()
            else:
                prev_kps_pixel = prev_kps_pixel * 0.7 + curr_pixel * 0.3
                
            hip_center = (prev_kps_pixel[11] + prev_kps_pixel[12]) / 2.0
            lpn_input = (prev_kps_pixel - hip_center).flatten()
            
            if len(joint_buffer) == 0: 
                joint_buffer = [lpn_input] * WINDOW_SIZE
            else:
                joint_buffer.append(lpn_input)
                if len(joint_buffer) > WINDOW_SIZE: joint_buffer.pop(0)
            
        # 2. TCN 추론 파이프라인
        if len(joint_buffer) == WINDOW_SIZE:
            seq = np.array(joint_buffer).reshape(1, 60, 26)
            
            # 💡 수정 완료: l_in_s -> ln_in_s, l_in_z -> ln_in_z
            l_in_q = np.clip(np.round(seq / ln_in_s + ln_in_z), -128, 127).astype(np.int8)
            ln_int.set_tensor(ln_in['index'], l_in_q)
            ln_int.invoke()
            l_out_f = (ln_int.get_tensor(ln_out['index']).astype(np.float32) - ln_out_z) * ln_out_s
            
            max_dist_3d = np.max(np.abs(l_out_f)) + 1e-6
            l_out_norm = l_out_f / max_dist_3d
            
            tcn_in_q = np.clip(np.round(l_out_norm / tcn_in_s + tcn_in_z), -128, 127).astype(np.int8)
            tcn_int.set_tensor(tcn_in['index'], tcn_in_q)
            tcn_int.invoke()
            res_f = (tcn_int.get_tensor(tcn_out['index']).astype(np.float32) - tcn_out_z) * tcn_out_s
            
            current_score = res_f.flatten()[0]
            video_scores.append(current_score)

        if current_score > FALL_THRESHOLD:
            status_text = "FALL DETECTED!"
            status_color = (0, 0, 255) 
        else:
            status_text = "NORMAL"
            status_color = (0, 255, 0) 
            
        cv2.rectangle(annotated_frame, (5, 5), (350, 75), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"STATUS: {status_text}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(annotated_frame, f"Score: {current_score:.4f}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out_writer.write(annotated_frame)

    cap.release()
    out_writer.release()

    if video_scores:
        video_scores.sort(reverse=True)
        max_score = np.mean(video_scores[:5]) 
        final_pred = 1 if max_score > FALL_THRESHOLD else 0
    else:
        max_score = 0.0
        final_pred = 0

    y_true_list.append(label)
    y_pred_list.append(final_pred)
    
    status = "✅ 정답" if label == final_pred else "❌ 오답"
    print(f"\r▶ [{filename}] MaxScore: {max_score:.4f} | 판정: {'FALL' if final_pred else 'NORMAL'} | {status}")

# ==========================================
# 📊 4. 최종 리포트
# ==========================================
print("\n" + "="*60)
print(f"🎉 렌더링 완료! 결과물은 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
if len(y_true_list) > 0:
    acc = accuracy_score(y_true_list, y_pred_list)
    rec = recall_score(y_true_list, y_pred_list, zero_division=0)
    print(f"📊 파이프라인 일관성 교정 결과 (관성 예측 + 시각화)")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - Recall:   {rec*100:.2f}%")
print("="*60)