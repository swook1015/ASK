import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. 데이터 로드 및 정규화
X_data = np.load('X_data.npy').astype(np.float32)
y_data = np.load('y_data.npy').astype(np.float32)

def normalize_pose(data):
    norm_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            frame = data[i, j, :]
            xs, ys, cs = frame[0::3], frame[1::3], frame[2::3]
            valid = cs > 0.1
            if np.any(valid):
                cx, cy = np.mean(xs[valid]), np.mean(ys[valid])
                mr = max(np.max(xs[valid]) - np.min(xs[valid]), np.max(ys[valid]) - np.min(ys[valid])) + 1e-6
                norm_data[i, j, 0::3] = (xs - cx) / mr
                norm_data[i, j, 1::3] = (ys - cy) / mr
                norm_data[i, j, 2::3] = cs
    return norm_data

X_final = normalize_pose(X_data)

# 2. 모델 설계
def build_model():
    model = models.Sequential([
        layers.Input(shape=(20, 51), batch_size=1, name='input'),
        layers.LSTM(8),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model

model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. 학습 (에폭 1로 구조만 테스트)
print("🚀 구조 검증용 학습 시작...")
model.fit(X_final, y_data, epochs=1, batch_size=1)

# 4. 🛠️ Edge TPU 양자화 (Concrete Function 우회법)
print("\n🛠️ Edge TPU 완전 정수 양자화 시작...")

# [핵심] 입구를 강제로 정의하여 함수로 추출
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 20, 51], model.inputs[0].dtype))

# 변환기를 Keras 모델이 아닌 Concrete Function으로 생성
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 대표 데이터셋 (범위를 더 넓게 잡기 위해 200개)
def representative_data_gen():
    for i in range(200):
        yield [X_final[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_data_gen

# 완전 정수 양자화 강제 설정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_lower_tensor_list_ops = True

try:
    tflite_model = converter.convert()
    with open('concrete_final_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("\n✅ 변환 성공! 로그를 확인하세요.")
except Exception as e:
    print(f"\n❌ 변환 실패: {e}")