import os
import numpy as np

# 아까 낱개 npy 파일들이 저장된 폴더 경로
npy_folder = './AI/dataset/nturgb+d_skeletons-npy/'
file_list = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

X_data = []
y_data = []

print("📦 데이터를 거대한 하나의 문제집으로 모으는 중...")

for file_name in file_list:
    file_path = os.path.join(npy_folder, file_name)
    data = np.load(file_path) # (60, 39)
    
    # NTU 데이터셋 기준: 파일명에 A043이 들어가면 '낙상'
    if 'A043' in file_name:
        label = 1 # 낙상
    else:
        label = 0 # 정상 (낙상 외의 모든 행동)
        
    X_data.append(data)
    y_data.append(label)

# 딥러닝이 먹을 수 있게 Numpy 배열로 최종 변환
X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

print("\n🎉 데이터셋 병합 완료!")
print(f"✅ X_train 형태 (문제집): {X_data.shape}") # 예: (56000, 60, 39)
print(f"✅ y_train 형태 (정답지): {y_data.shape}") # 예: (56000,)
print(f"🚨 총 데이터 중 낙상(A043) 데이터 개수: {np.sum(y_data)}개")

# TCN 학습을 위해 최종 파일로 저장
np.save('ntu_X_train_60frames.npy', X_data)
np.save('ntu_y_train_60frames.npy', y_data)