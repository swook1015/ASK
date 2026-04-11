import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

# 💡 1. 경로 설정
TFLITE_PATH = './AI/models/ln/new_lpn_cv_60_legacy.tflite'
X_TEST_PATH = './AI/dataset/CV_Split/X_cv_test.npy'

# 💡 2. 데이터 로드 및 2D/3D 분리 (전체 데이터 사용)
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

# 💡 3. TFLite 인터프리터 준비
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# 양자화 파라미터 (INT8인 경우 필요)
in_scale, in_zero = input_details['quantization']
out_scale, out_zero = output_details['quantization']

# 💡 4. TFLite 추론 루프
print(f"\n🚀 TFLite 모델({os.path.basename(TFLITE_PATH)}) 추론 시작...")
y_pred_list = []

for i in tqdm(range(len(X_test)), desc="TFLite Predicting"):
    inp = X_test[i:i+1] # (1, 60, 26)
    
    # 양자화 적용 (INT8 모델인 경우 자동 계산, FP32면 scale=1.0, zero=0)
    if in_scale != 0:
        inp = np.clip(np.round(inp / in_scale + in_zero), -128, 127).astype(np.int8)
    
    interpreter.set_tensor(input_details['index'], inp)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details['index'])
    
    # 역양자화 (실수값 복원)
    if out_scale != 0:
        output = (output.astype(np.float32) - out_zero) * out_scale
    
    y_pred_list.append(output)

Y_pred = np.vstack(y_pred_list) # (N, 60, 39)

# 💡 5. 성능 지표 계산
print("\n" + "="*50)
print(f"📊 [TFLite 전수 검증 결과]")

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

print(f"📏 최종 MPJPE: {mpjpe:.8f}")
print(f"   (이 값이 낮을수록 실시간 영상에서 뼈대 복원을 잘한다는 뜻!)")
print("="*50)