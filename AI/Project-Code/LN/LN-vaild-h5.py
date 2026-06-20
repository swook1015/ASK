import numpy as np
import tensorflow as tf
import os

# 💡 1. 경로 설정 (H5 파일 경로)
H5_PATH = './AI/models/ln/new_lpn_cv_60_legacy.h5' 
X_TEST_PATH = './AI/dataset/CV_Split/X_cv_test.npy'

# ---------------------------------------------------------
# 🛠️ [Keras 3 패치] 레이어 호환성 문제 해결 클래스
# ---------------------------------------------------------
class PatchedBatchNormalization(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        if 'axis' in config and isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return super(PatchedBatchNormalization, cls).from_config(config)

# 💡 2. 데이터 로드 및 2D/3D 분리
def load_lpn_data(path):
    print(f"📦 테스트 데이터 로딩 중: {os.path.basename(path)}")
    raw_data = np.load(path).astype(np.float32)
    N, T, _ = raw_data.shape
    
    # [Y] 정답지: 3D 전체 (x, y, z)
    Y_true = raw_data
    
    # [X] 입력값: 2D만 추출 (x, y) -> (N, 60, 26)
    data_reshaped = raw_data.reshape(N, T, 13, 3)
    X_input = data_reshaped[:, :, :, 0:2].reshape(N, T, 26)
    
    return X_input, Y_true

X_test, Y_test = load_lpn_data(X_TEST_PATH)

# 💡 3. H5 모델 로드 (Keras 3 호환 모드)
print(f"\n📦 H5 모델({os.path.basename(H5_PATH)}) 로드 중...")
custom_objs = {
    'Dense': tf.keras.layers.Dense,
    'Activation': tf.keras.layers.Activation,
    'BatchNormalization': PatchedBatchNormalization,
    'Dropout': tf.keras.layers.Dropout,
    'ReLU': tf.keras.layers.ReLU,
    'TimeDistributed': tf.keras.layers.TimeDistributed,
    'InputLayer': tf.keras.layers.InputLayer,
    'Add': tf.keras.layers.Add,
    'Multiply': tf.keras.layers.Multiply,
    'Reshape': tf.keras.layers.Reshape,
    'Concatenate': tf.keras.layers.Concatenate
}

model_h5 = tf.keras.models.load_model(H5_PATH, compile=False, custom_objects=custom_objs)

# 💡 4. H5 추론 (for문 없이 한 번에 배치로 밀어넣어 초고속 처리)
print(f"\n🚀 H5 모델 추론 시작 (배치 처리)...")
Y_pred = model_h5.predict(X_test, batch_size=1024, verbose=1)

# 💡 5. 성능 지표 계산
print("\n" + "="*50)
print(f"📊 [H5 원본 모델 전수 검증 결과]")

# MSE / MAE
mse = np.mean((Y_test - Y_pred)**2)
mae = np.mean(np.abs(Y_test - Y_pred))
print(f"  - Loss (MSE): {mse:.8f}")
print(f"  - MAE       : {mae:.8f}")

# MPJPE 계산
y_true_3d = Y_test.reshape(-1, 60, 13, 3)
y_pred_3d = Y_pred.reshape(-1, 60, 13, 3)

# 관절별 유클리드 거리 측정
dist = np.sqrt(np.sum((y_true_3d - y_pred_3d)**2, axis=-1)) # (N, 60, 13)
mpjpe = np.mean(dist)

# 형이 보던 원래 미터(m) 단위 출력 유지 + 보고서용 mm 단위 동시 출력
print(f"📏 최종 MPJPE: {mpjpe:.8f} m ({mpjpe*1000:.2f} mm)")
print(f"   (이 값이 낮을수록 실시간 영상에서 뼈대 복원을 잘한다는 뜻!)")
print("="*50)