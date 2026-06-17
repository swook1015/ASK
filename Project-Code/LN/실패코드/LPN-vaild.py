import numpy as np
import tensorflow as tf
import os

# 💡 Keras 2 Legacy 설정
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 1. 경로 설정
model_path = './AI/models/lpn/new_lpn_cv_60_legacy.tflite'
x_test_path = './AI/dataset/CV_Split/X_cv_test.npy'
y_test_path = './AI/dataset/CV_Split/y_cv_test.npy'

# 2. 데이터 로드 (마지막 10%만 검증용으로 사용했다고 가정)
def get_val_data(x_path, y_path, seq_len=60):
    X = np.load(x_path).astype('float32')
    Y = np.load(y_path).astype('float32')
    
    num_samples = len(X) // seq_len
    X = X[:num_samples * seq_len].reshape(-1, seq_len, 26)
    Y = Y[:num_samples * seq_len].reshape(-1, seq_len, 39)
    
    # 학습 때 validation_split=0.1을 썼으므로 뒤쪽 10% 추출
    split_idx = int(len(X) * 0.9)
    return X[split_idx:], Y[split_idx:]

X_val, Y_val = get_val_data(x_test_path, y_test_path)

# 3. 모델 평가
model = tf.keras.models.load_model(model_path)
results = model.evaluate(X_val, Y_val, batch_size=64)
print(f"\n📊 검증 데이터 결과 -> Loss(MSE): {results[0]:.6f}, MAE: {results[1]:.6f}")

# 4. 개별 샘플 테스트 (MPJPE 계산)
y_pred = model.predict(X_val[:100], verbose=0) # 100개 샘플만 테스트
error = np.sqrt(np.sum((Y_val[:100] - y_pred)**2, axis=-1)) # 유클리드 거리
print(f"📏 평균 관절 위치 오차(MPJPE): {np.mean(error):.6f}")