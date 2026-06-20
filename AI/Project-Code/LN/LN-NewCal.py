import numpy as np
import os

# 💡 1. 경로 설정 (방금 새로 뽑은 데이터 경로)
TRAIN_NPY_PATH = './AI/dataset/CV_Split/X_cv_train.npy'
SAVE_CALIB_PATH = './AI/dataset/calibration_ln_200_samples.npy'

# 대표 데이터 개수 (보통 100~500개면 충분)
TARGET_SAMPLES = 200 

def generate_ln_calibration_data():
    print(f"📦 LN 캘리브레이션 데이터 생성 시작...")

    if not os.path.exists(TRAIN_NPY_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {TRAIN_NPY_PATH}")
        return

    # 1. 3D 정답지 데이터 로드 (N, 60, 39)
    raw_data = np.load(TRAIN_NPY_PATH).astype(np.float32)
    N, T, _ = raw_data.shape
    print(f"✅ 원본 데이터 로드 완료: {raw_data.shape}")

    # 2. 2D 입력 데이터로 변환 (N, 60, 13, 3) -> X, Y만 추출 -> (N, 60, 26)
    # LPN 모델이 학습할 때 보던 '입력' 형태로 정확히 맞춰줍니다.
    data_reshaped = raw_data.reshape(N, T, 13, 3)
    X_2d_input = data_reshaped[:, :, :, 0:2].reshape(N, T, 26)
    print(f"✅ 2D 입력 규격 변환 완료: {X_2d_input.shape}")

    # 3. 랜덤 샘플링 (대표 데이터 200개 추출)
    np.random.seed(42) # 매번 같은 데이터가 뽑히도록 시드 고정
    indices = np.random.choice(N, min(TARGET_SAMPLES, N), replace=False)
    calib_data = X_2d_input[indices]

    # 4. 파일 저장
    os.makedirs(os.path.dirname(SAVE_CALIB_PATH), exist_ok=True)
    np.save(SAVE_CALIB_PATH, calib_data)
    
    print("\n" + "="*50)
    print(f"🎉 LN 캘리브레이션 데이터 저장 완료!")
    print(f"💾 저장 경로: {SAVE_CALIB_PATH}")
    print(f"📏 최종 규격: {calib_data.shape} (자료형: {calib_data.dtype})")
    print("💡 이제 새로 학습된 H5 모델과 이 npy 파일을 코랩으로 가져가서 양자화하세요!")
    print("="*50)

if __name__ == "__main__":
    generate_ln_calibration_data()