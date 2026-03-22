import numpy as np
import os
import glob
import random

# ==========================================
# 1. 셔플된 NTU 3D 데이터 복원 (X_ntu, y_ntu)
# ==========================================
print("📦 셔플된 NTU 3D 데이터 복원 중...")
ntu_3d_path = './AI/dataset/LPN-label/ntu_lpn_target_3d_shuffled.npy'
skeleton_dir = './AI/dataset/nturgb+d_skeletons-npy/'

raw_ntu = np.load(ntu_3d_path) # (N*60, 39)
num_ntu = raw_ntu.shape[0] // 60
X_ntu = raw_ntu.reshape(num_ntu, 60, 39)

# 💡 [수정된 부분] X를 만들 때와 완벽하게 동일한 조건으로 필터링하며 라벨 생성
npy_files = sorted(glob.glob(os.path.join(skeleton_dir, "*.skeleton.npy")))
random.seed(42) 
random.shuffle(npy_files)

y_ntu_list = []
print("🔍 라벨 무결성 검사 중... (시간이 조금 걸릴 수 있습니다)")
for f in npy_files:
    try:
        # X를 만들 때처럼 실제로 열어서 60프레임짜리가 맞는지 확인
        temp_data = np.load(f)
        if temp_data.shape == (60, 1, 39):
            y_ntu_list.append(1 if 'A043' in f else 0)
    except:
        continue

y_ntu = np.array(y_ntu_list).reshape(-1, 1)

# X와 y의 개수가 정확히 일치하는지 최종 확인
assert X_ntu.shape[0] == y_ntu.shape[0], f"🚨 갯수 불일치! X: {X_ntu.shape[0]}, y: {y_ntu.shape[0]}"
print(f"✅ NTU 복원 및 라벨 생성 완벽 일치! (샘플 수: {X_ntu.shape[0]})")

# ==========================================
# 2. URFD 3D 데이터 로드 (X_urfd, y_urfd)
# ==========================================
print("📦 URFD 3D 데이터 로드 중...")
X_urfd = np.load('./AI/dataset/urfd_processed/urfd_3d_train_X.npy') # (N, 60, 39)
y_urfd = np.load('./AI/dataset/urfd_processed/urfd_3d_train_y.npy') # (N, 1)

# ==========================================
# 3. 최종 병합 및 저장
# ==========================================
X_combined = np.concatenate([X_ntu, X_urfd], axis=0)
y_combined = np.concatenate([y_ntu, y_urfd], axis=0)

# 마지막으로 전체를 한 번 더 섞어줌 (NTU와 URFD가 섞이도록)
indices = np.arange(X_combined.shape[0])
np.random.shuffle(indices)
X_final = X_combined[indices]
y_final = y_combined[indices]

np.save('./combined_train_X.npy', X_final)
np.save('./combined_train_y.npy', y_final)

print(f"✅ 최종 병합 완료! 모양: X={X_final.shape}, y={y_final.shape}")