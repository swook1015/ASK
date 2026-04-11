import numpy as np
import cv2
import glob

# 1. 학습에 사용했던 실제 낙상 이미지 100장 가져오기 (경로 주의)
img_paths = glob.glob('C:/Users/pcroom2/Desktop/ASK/AI/dataset/AIP.v2i.yolov8/train/images/*.jpg')[:100]

calib_data = []
for path in img_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) # 학습한 224 해상도와 동일하게
    img = img.astype(np.float32) / 255.0 # 0~1 정규화
    img = np.transpose(img, (2, 0, 1)) # NCHW 포맷으로 변경
    calib_data.append(img)

calib_data = np.array(calib_data, dtype=np.float32)
np.save('calibration_data.npy', calib_data)
print("✅ 캘리브레이션 데이터 생성 완료:", calib_data.shape) # (100, 3, 224, 224) 확인