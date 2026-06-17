import numpy as np
import cv2
import glob
import os
from ultralytics import YOLO

# 1. Ultralytics를 속일 가짜 파일명 (다운로드 스킵 유도)
fake_filename = 'calibration_image_sample_data_20x128x128x3_float32.npy'

# 2. 진짜 낙상 데이터 100장으로 완벽한 224x224 캘리브레이션 파일 생성
img_paths = glob.glob('C:/Users/pcroom2/Desktop/ASK/AI/dataset/AIP.v2i.yolov8/train/images/*.jpg')[:100]
calib_data = []
for path in img_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # NCHW 포맷
    calib_data.append(img)

# 트로이 목마 파일 덮어쓰기
calib_data = np.array(calib_data, dtype=np.float32)
np.save(fake_filename, calib_data)
print(f"✅ 속임수용 완벽한 캘리브레이션 데이터 준비 완료: {calib_data.shape}")

# 3. 자동 변환 실행 (에러 없이 우리의 완벽한 데이터를 먹고 변환됩니다!)
model = YOLO(r'C:\Users\pcroom2\Desktop\ASK\runs\pose\v8n_pose_fall_relu_fixed\weights\best.pt')
model.export(
    format='tflite', 
    int8=True, 
    imgsz=224, 
    data=r'C:\Users\pcroom2\Desktop\ASK\AI\dataset\AIP.v2i.yolov8\data.yaml'
)