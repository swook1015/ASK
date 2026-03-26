import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import psutil

# Keras 2(Legacy) 엔진 강제 사용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# ⚙️ 1. 모델 로드 및 설정
# ==========================================
print("🚀 [1] 모델 로드 중...")
yolo_model = YOLO('AI/models/yolo/yolov8n-pose.pt') 
lpn_model = tf.keras.models.load_model('lpn_remaster_60_legacy.h5')
tcn_model = tf.keras.models.load_model('tcn_fall_detector_stride_best.h5')

VIDEO_PATH = '1.mp4' 
OUTPUT_PATH = '1result_video_optimized.mp4' 

cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# LPN/TCN 입력에 필요한 13개 관절 인덱스
KEEP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# 13개 관절 기준 연결선 (거미줄 방지)
NEW_EDGES = [
    (0,1), (0,2), (1,2),          # 머리-어깨
    (1,3), (3,5), (2,4), (4,6),   # 팔
    (1,7), (2,8), (7,8),          # 몸통
    (7,9), (9,11), (8,10), (10,12)# 다리
]

# 실시간 추론을 시뮬레이션하기 위한 60프레임 버퍼
joint_buffer = []

print(f"⏳ [2] 영상 처리 시작: {VIDEO_PATH} ({width}x{height}, {fps:.1f}fps)")

# ==========================================
# 🧠 2. 메인 처리 루프
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8-pose 추론 (정사각형 리사이즈 없이 원본 비율 유지 시도)
    results = yolo_model(frame, verbose=False)
    
    # 1. 관절 좌표 추출
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # shape: (17, 3) -> [x, y, conf]
        kpts = results[0].keypoints.data[0].cpu().numpy()
    else:
        kpts = np.zeros((17, 3))

    # 2. LPN/TCN 입력을 위한 데이터 정규화 (핵심: [-1, 1] 범위)
    # 구형 get_movenet_norm_coords 제거
    current_frame_2d = []
    for idx in KEEP_INDICES:
        x, y, _ = kpts[idx]
        # 문서 가이드라인 준수: [0, 1] 변환 후 [-1, 1]로 매핑
        norm_x = (x / width) * 2.0 - 1.0
        norm_y = (y / height) * 2.0 - 1.0
        current_frame_2d.extend([norm_x, norm_y])

    # 3. 슬라이딩 윈도우 버퍼 업데이트
    joint_buffer.append(current_frame_2d)
    if len(joint_buffer) > 60:
        joint_buffer.pop(0)

    # 4. 낙상 추론 (60프레임이 찼을 때만 실행)
    fall_prob = 0.0
    if len(joint_buffer) == 60:
        seq_2d = np.array(joint_buffer, dtype=np.float32).reshape(1, 60, 26)
        seq_3d = lpn_model.predict(seq_2d, verbose=0)
        fall_prob = tcn_model.predict(seq_3d, verbose=0)[0][0]

    # ==========================================
    # 🖼️ 3. 시각화 (Annotating)
    # ==========================================
    draw_frame = frame.copy()
    
    # 상태 텍스트 설정
    if fall_prob > 0.85:
        status_text = f"FALL DETECTED! ({fall_prob*100:.1f}%)"
        color = (0, 0, 255) # 빨간색
    else:
        status_text = f"NORMAL ({fall_prob*100:.1f}%)"
        color = (255, 0, 0) # 파란색

    # 13개 관절 뼈대 그리기
    for edge in NEW_EDGES:
        p1_idx, p2_idx = edge
        # kpts는 17개 관절이므로 KEEP_INDICES의 실제 인덱스를 사용
        p1_orig = KEEP_INDICES[p1_idx]
        p2_orig = KEEP_INDICES[p2_idx]
        
        x1, y1, c1 = kpts[p1_orig]
        x2, y2, c2 = kpts[p2_orig]
        
        if c1 > 0.4 and c2 > 0.4: # 신뢰도 높은 선만 그리기
            cv2.line(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 관절점 그리기
    for idx in KEEP_INDICES:
        x, y, c = kpts[idx]
        if c > 0.4:
            cv2.circle(draw_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # UI 텍스트 오버레이
    cv2.rectangle(draw_frame, (20, 15), (550, 70), (0, 0, 0), -1)
    cv2.putText(draw_frame, status_text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    out.write(draw_frame)

cap.release()
out.release()

print("\n" + "="*50)
print(f"✅ 개선된 결과 영상 저장 완료: {OUTPUT_PATH}")
print("="*50)