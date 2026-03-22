import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO  # YOLO 라이브러리 추가

# Keras 2(Legacy) 엔진 강제 사용 (호환성 유지)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

print("🚀 [1] 로컬 모델 3종 세트 로드 중...")
# MoveNet 대신 YOLOv8-pose 모델 로드 (업로드된 가중치 파일 사용)
yolo_model = YOLO('AI/models/yolo/yolov8n-pose.pt') 
lpn_model = tf.keras.models.load_model('lpn_remaster_60_legacy.h5')
tcn_model = tf.keras.models.load_model('tcn_fall_detector_stride_best.h5')

print("🎥 [2] 영상 로드 및 시각화 준비")
VIDEO_PATH = '3.mp4' 
OUTPUT_PATH = 'result_video.mp4' 

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frames_2d = []
annotated_frames = [] 

edges = [
    (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# 💡 추가됨: YOLO의 절대 좌표를 기존 MoveNet의 256x256 패딩 기준 정규화 좌표(0~1)로 변환
# (이후의 lpn_model이 기존과 동일한 형태의 데이터를 받도록 호환성 유지)
def get_movenet_norm_coords(orig_x, orig_y, w, h):
    target = 256.0
    scale = target / max(w, h)
    pad_x = (target - (w * scale)) / 2.0
    pad_y = (target - (h * scale)) / 2.0
    
    x_norm = (orig_x * scale + pad_x) / target
    y_norm = (orig_y * scale + pad_y) / target
    return x_norm, y_norm

print("⏳ [3] 프레임 추출 및 뼈대 그리기 진행 중 (영상 끝까지)...")
keep_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8-pose 추론 실행
    results = yolo_model(frame, verbose=False)

    # 관절 좌표 추출 (사람이 감지된 경우 첫 번째 사람 기준)
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # shape: (17, 3) -> [x, y, confidence]
        kpts = results[0].keypoints.data[0].cpu().numpy() 
    else:
        kpts = np.zeros((17, 3))

    keypoints_13 = []
    for idx in keep_indices:
        x, y, conf = kpts[idx]
        # 모델 입력을 위해 좌표 정규화 진행
        x_norm, y_norm = get_movenet_norm_coords(x, y, width, height)
        keypoints_13.extend([x_norm, y_norm])
    frames_2d.append(keypoints_13)

    draw_frame = frame.copy()
    
    # 뼈대 그리기 (YOLO는 절대 픽셀 좌표를 반환하므로 복잡한 매핑 없이 바로 사용)
    for edge in edges:
        p1, p2 = edge
        x1, y1, s1 = kpts[p1]
        x2, y2, s2 = kpts[p2]
        if s1 > 0.2 and s2 > 0.2:
            cv2.line(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 관절점 그리기
    for idx in keep_indices:
        x, y, s = kpts[idx]
        if s > 0.2:
            cv2.circle(draw_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    annotated_frames.append(draw_frame)

cap.release()

total_frames = len(frames_2d)
if total_frames == 0:
    print("❌ 프레임 추출 실패")
    exit()

print(f"✅ 원본 영상 총 {total_frames} 프레임 추출 완료.")

print("🏗️ [4] AI 입력을 위한 60프레임 샘플링 및 모델 추론")
if total_frames > 60:
    indices = np.linspace(0, total_frames - 1, 60).astype(int)
    sampled_2d = [frames_2d[i] for i in indices]
elif total_frames < 60:
    sampled_2d = frames_2d.copy()
    while len(sampled_2d) < 60:
        sampled_2d.append(sampled_2d[-1])
else:
    sampled_2d = frames_2d.copy()

sequence_2d = np.array(sampled_2d, dtype=np.float32).reshape(1, 60, 26)
sequence_3d = lpn_model.predict(sequence_2d, verbose=0)
fall_prob = tcn_model.predict(sequence_3d, verbose=0)[0][0]

print("📼 [5] 영상 합성 및 저장 중...")
if fall_prob > 0.85:
    result_text = f"FALL DETECTED! ({fall_prob*100:.1f}%)"
    color = (0, 0, 255) 
else:
    result_text = f"NORMAL ({fall_prob*100:.1f}%)"
    color = (255, 0, 0) 

for frm in annotated_frames:
    cv2.putText(frm, result_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
    out.write(frm)

out.release()
print("\n" + "="*50)
print(f"✅ 전체 길이가 담긴 결과 영상 저장 완료: {OUTPUT_PATH}")
print("="*50)