import numpy as np

print("=== 1. 데이터 불러오기 ===")
X_ntu_train = np.load('./AI/dataset/LPN-train/ntu_lpn_target_3d_final.npy') 
y_ntu_train = np.load('./AI/dataset/LPN-train/')

X_urfd_train = np.load('./AI/dataset/urfd_processed/urfd_3d_train_X.npy')
y_urfd_train = np.load('./AI/dataset/urfd_processed/urfd_3d_train_y.npy')

print("=== 2. 규격 맞추기 (에러 해결의 핵심!) ===")
# 1) NTU X 데이터 차원 축소
if len(X_ntu_train.shape) == 4:
    X_ntu_train = np.squeeze(X_ntu_train, axis=2)

# 2) NTU y 데이터 라벨 변환
if len(y_ntu_train.shape) == 2:
    y_ntu_train = np.argmax(y_ntu_train, axis=1)

# 3) URFD y 데이터 규격 맞추기 (2차원 -> 1차원)
if len(y_urfd_train.shape) == 2:
    y_urfd_train = y_urfd_train.flatten()

print("=== 3. 데이터 합치기 및 전체 셔플 ===")
# 일단 두 데이터를 합칩니다.
X_train = np.concatenate([X_ntu_train, X_urfd_train], axis=0)
y_train = np.concatenate([y_ntu_train, y_urfd_train], axis=0)

# 🔥 [핵심] 통합 인덱스를 생성하여 전체를 무작위로 뒤섞습니다.
# X와 y의 인덱스 순서가 어긋나지 않도록 동일한 indices 배열을 사용합니다.
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]

print(f"🔥 최종 학습 데이터 모양: X={X_train.shape}, y={y_train.shape}")

print("=== 4. 파일 저장 ===")
np.save('./combined_train_X.npy', X_train)
np.save('./combined_train_y.npy', y_train)
print("💾 병합 및 전체 셔플이 완료된 데이터가 저장되었습니다!")