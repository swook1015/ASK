import numpy as np

# 1. 1D-TCN 학습에 사용된 정규화 완료 3D 데이터 로드 (N, 60, 39)
# CV-npy.py로 만든 정적 스케일링(480.0) 및 Y축 반전이 적용된 그 데이터입니다.
tcn_3d_data = np.load('./AI/dataset/CV_Split/X_cv_train.npy').astype(np.float32)

# 2. 200개 랜덤 샘플링 (양자화 최적화 용량)
TARGET_SAMPLES = 200
np.random.seed(42)
calib = tcn_3d_data[np.random.choice(len(tcn_3d_data), TARGET_SAMPLES, replace=False)]

# 3. 캘리브레이션 파일 저장
save_path = f'calibration_tcn_{TARGET_SAMPLES}_samples.npy'
np.save(save_path, calib)

print(f"✅ 1D-TCN 전용 캘리브레이션 데이터 생성 완료: {calib.shape}")