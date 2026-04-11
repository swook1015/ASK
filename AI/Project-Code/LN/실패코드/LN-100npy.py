import numpy as np

# 1. 이미 센터링 및 정규화가 완료된 학습용 3D 데이터 로드 (N, 60, 39)
ntu_3d = np.load('./AI/dataset/CV_Split/X_cv_train.npy').astype(np.float32)

# 2. 13관절의 (X, Y, Z) 중 Z축을 제외하고 (X, Y)만 추출
reshaped = ntu_3d.reshape(-1, 60, 13, 3)

# (주의: CV-npy.py에서 이미 골반 기준으로 영점 조절이 끝났으므로 바로 자르기만 하면 됩니다)
ntu_2d = reshaped[:, :, :, :2].reshape(-1, 60, 26)

# 3. 200개 랜덤 샘플링 (원하시면 500으로 변경 가능)
TARGET_SAMPLES = 200
np.random.seed(42)
calib = ntu_2d[np.random.choice(len(ntu_2d), TARGET_SAMPLES, replace=False)]

# 4. 캘리브레이션 파일 저장
save_path = f'calibration_lpn_{TARGET_SAMPLES}_samples.npy'
np.save(save_path, calib)

print(f"✅ 초간단 NTU 기반 캘리브레이션 데이터 생성 완료: {calib.shape}")