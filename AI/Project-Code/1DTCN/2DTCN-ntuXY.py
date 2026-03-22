import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# 낱개 npy 파일 경로
npy_folder = './AI/dataset/LPN-train/ntu_lpn_target_3d_final.npy'
file_list = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

X_list = []
y_list = []

for file_name in file_list:
    file_path = os.path.join(npy_folder, file_name)
    data = np.load(file_path) # 로드 시 (60, 1, 39) 형태 확인
    
    # 1. 차원 강제 고정 (60, 1, 39)
    if data.shape != (60, 1, 39):
        data = data.reshape(60, 1, 39)
        
    X_list.append(data)
    y_list.append(1 if 'A043' in file_name else 0)

# 2. Numpy 배열 변환 및 One-hot 인코딩
X_data = np.array(X_list, dtype=np.float32)
y_data = to_categorical(np.array(y_list), num_classes=2) # 0 -> [1,0], 1 -> [0,1]

# 3. 최종 저장
np.save('./AI/dataset/2D-TCN/ntu_X_60frames_coral.npy', X_data)
np.save('./AI/dataset/2D-TCN/ntu_y_60frames_onehot.npy', y_data)

print(f"✅ 병합 완료! X 형태: {X_data.shape}, y 형태: {y_data.shape}")