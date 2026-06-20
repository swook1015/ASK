import os
import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# ==========================================
# ⚙️ 1. 설정 및 경로
# ==========================================
INPUT_VIDEO = './input/fall1.avi'
OUTPUT_DIR = './output_debug'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, 'result_fall1.avi')

POSE_MODEL = './yolo/yolo8n-pose_full_integer_quant_edgetpu.tflite'
LPN_MODEL  = './model/0408new_lpn_cv_60_legacy_edgetpu.tflite'
TCN_MODEL  = './model/0408tcn_8layer_pure_ntu_edgetpu.tflite'

WINDOW_SIZE = 60
FALL_THRESHOLD = 0.9 

# PC 버전과 100% 동일한 연결맵 (시각화용)
SKELETON_MAP = [
    (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (0, 7), (0, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)
]

# ==========================================
# 🧠 2. 모델 로드 및 양자화 파라미터 준비
# ==========================================
print("🚀 코랄 보드 전용 시각화 및 지연 시간 측정 파이프라인 로드 중...")
pose_int = make_interpreter(POSE_MODEL); pose_int.allocate_tensors()
ln_int   = make_interpreter(LPN_MODEL);  ln_int.allocate_tensors()
tcn_int  = make_interpreter(TCN_MODEL);  tcn_int.allocate_tensors()

p_in_det = pose_int.get_input_details()[0]
H_IN, W_IN = p_in_det['shape'][1], p_in_det['shape'][2]
p_out_s, p_out_z = pose_int.get_output_details()[0]['quantization']

ln_in_s, ln_in_z = ln_int.get_input_details()[0]['quantization']
ln_out_s, ln_out_z = ln_int.get_output_details()[0]['quantization']
tcn_in_s, tcn_in_z = tcn_int.get_input_details()[0]['quantization']
tcn_out_s, tcn_out_z = tcn_int.get_output_details()[0]['quantization']

def center_poses_2d_final(X_2d_window, global_max_dist):
    root = X_2d_window[:, 0:1, :]
    centered = X_2d_window - root
    centered[:, :, 1] = -centered[:, :, 1] 
    normalized = centered / global_max_dist
    return normalized.reshape(1, WINDOW_SIZE, 26).astype(np.float32)

# ==========================================
# 🎥 3. [1차 패스] 영상 전체 기준 스케일 고정
# ==========================================
print("📏 1차 패스: 글로벌 스케일 추출 중...")
cap = cv2.VideoCapture(INPUT_VIDEO)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

raw_kpts_list = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    img_input = cv2.resize(frame, (W_IN, H_IN))
    common.set_input(pose_int, cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
    pose_int.invoke()
    
    raw_out = np.squeeze(pose_int.get_tensor(pose_int.get_output_details()[0]['index']))
    best_idx = np.argmax(raw_out[4, :])
    conf = (raw_out[4, best_idx] - p_out_z) * p_out_s
    
    if conf > 0.05:
        raw_kpts = (raw_out[5:56, best_idx].astype(np.float32) - p_out_z) * p_out_s
        kpts_17 = raw_kpts.reshape(17, 3)
        curr_kpts = kpts_17[:, :2] # x, y 좌표만 추출
        
        # 원본 비율 복원
        curr_kpts[:, 0] *= (width / W_IN)
        curr_kpts[:, 1] *= (height / H_IN)
        
        # 🚨 [버그 완벽 수정] 발목이 아닌 11(왼쪽 골반), 12(오른쪽 골반) 인덱스를 정확히 타겟팅!!
        hip_center = (curr_kpts[11] + curr_kpts[12]) / 2.0
        kpts_13_input = np.vstack((hip_center, curr_kpts[5:17]))
        raw_kpts_list.append(kpts_13_input)

if raw_kpts_list:
    temp_kpts = np.array(raw_kpts_list)
    temp_root = temp_kpts[:, 0:1, :]
    temp_centered = temp_kpts - temp_root
    global_max_dist = np.max(np.abs(temp_centered)) + 1e-6
    last_valid_kpts = raw_kpts_list[0]
else:
    global_max_dist = 1.0
    last_valid_kpts = np.zeros((13, 2))

# ==========================================
# 📊 4. [2차 패스] 실제 추론, 로깅 및 시각화
# ==========================================
print(f"🎬 2차 패스: 추론 및 시각화 시작! 저장경로: {OUTPUT_VIDEO}")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

joint_buffer = [last_valid_kpts] * (WINDOW_SIZE - 1)
max_score = 0.0

print("-" * 75)
print(f"{'Frame':<8} | {'YOLO (ms)':<12} | {'LPN+TCN (ms)':<15} | {'Total (ms)':<12} | {'Status'}")
print("-" * 75)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    loop_start = time.perf_counter()
    ann_frame = frame.copy()
    
    # --- [YOLO 추론] ---
    yolo_start = time.perf_counter()
    img_input = cv2.resize(frame, (W_IN, H_IN))
    common.set_input(pose_int, cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
    pose_int.invoke()
    
    raw_out = np.squeeze(pose_int.get_tensor(pose_int.get_output_details()[0]['index']))
    best_idx = np.argmax(raw_out[4, :])
    conf = (raw_out[4, best_idx] - p_out_z) * p_out_s
    yolo_ms = (time.perf_counter() - yolo_start) * 1000

    current_frame_score = 0.0
    classifier_ms = 0.0

    if conf > 0.05:
        raw_kpts = (raw_out[5:56, best_idx].astype(np.float32) - p_out_z) * p_out_s
        kpts_17 = raw_kpts.reshape(17, 3)
        curr_kpts = kpts_17[:, :2]
        
        curr_kpts[:, 0] *= (width / W_IN)
        curr_kpts[:, 1] *= (height / H_IN)
        
        # 🚨 [버그 완벽 수정]
        hip_center = (curr_kpts[11] + curr_kpts[12]) / 2.0
        kpts_13_input = np.vstack((hip_center, curr_kpts[5:17]))
        
        last_valid_kpts = kpts_13_input 
        color = (0, 255, 255)
    else:
        kpts_13_input = last_valid_kpts
        color = (128, 128, 128)

    joint_buffer.append(kpts_13_input)
    if len(joint_buffer) > WINDOW_SIZE: 
        joint_buffer.pop(0)

    # 💡 뼈대 그리기
    draw_kpts = np.copy(kpts_13_input)
    for connection in SKELETON_MAP:
        pt1 = tuple(draw_kpts[connection[0]].astype(int))
        pt2 = tuple(draw_kpts[connection[1]].astype(int))
        cv2.line(ann_frame, pt1, pt2, color, 2)
    for kp in draw_kpts:
        cv2.circle(ann_frame, tuple(kp.astype(int)), 4, (255, 0, 0), -1)

    # --- [LPN + TCN 추론] ---
    if len(joint_buffer) == WINDOW_SIZE:
        classifier_start = time.perf_counter()
        
        input_2d = center_poses_2d_final(np.array(joint_buffer), global_max_dist)
        input_2d_q = np.clip(np.round(input_2d / ln_in_s + ln_in_z), -128, 127).astype(np.int8)
        
        common.set_input(ln_int, input_2d_q); ln_int.invoke()
        ln_out_f = (ln_int.get_tensor(ln_int.get_output_details()[0]['index']).astype(np.float32) - ln_out_z) * ln_out_s
        
        tcn_in_q = np.clip(np.round(ln_out_f / tcn_in_s + tcn_in_z), -128, 127).astype(np.int8)
        common.set_input(tcn_int, tcn_in_q); tcn_int.invoke()
        
        current_frame_score = (tcn_int.get_tensor(tcn_int.get_output_details()[0]['index']).astype(np.float32) - tcn_out_z) * tcn_out_s
        current_frame_score = current_frame_score.flatten()[0]
        
        if current_frame_score > max_score: max_score = current_frame_score
        classifier_ms = (time.perf_counter() - classifier_start) * 1000

        if current_frame_score > FALL_THRESHOLD:
            status = "FALL!!"
            text_color = (0, 0, 255)
        else:
            status = "NORMAL" if conf > 0.05 else "LOST TARGET (NORMAL)"
            text_color = color
    
    total_ms = (time.perf_counter() - loop_start) * 1000
    print(f"{frame_idx:<8} | {yolo_ms:<12.1f} | {classifier_ms:<15.1f} | {total_ms:<12.1f} | {status} ({current_frame_score:.2f})")

    cv2.putText(ann_frame, f"{status} ({current_frame_score:.2f})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
    out_writer.write(ann_frame)

cap.release()
out_writer.release()
print("-" * 75)
print(f"✅ 영상 렌더링 완료! 최대 낙상 스코어: {max_score:.4f}")
print(f"💾 확인 경로: {OUTPUT_VIDEO}")