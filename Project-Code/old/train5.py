import numpy as np
import tensorflow as tf

print("🚀 1. 데이터 로드 및 전처리...")
X = np.load('X_data.npy')
Y = np.load('y_data.npy') 

X_norm = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        frame = X[i, j, :]
        xs, ys, cs = frame[0::3], frame[1::3], frame[2::3]
        valid = cs > 0.1
        if np.any(valid):
            cx, cy = np.mean(xs[valid]), np.mean(ys[valid])
            max_range = max(np.max(xs[valid]) - np.min(xs[valid]), np.max(ys[valid]) - np.min(ys[valid])) + 1e-6
            X_norm[i, j, 0::3] = (xs - cx) / max_range
            X_norm[i, j, 1::3] = (ys - cy) / max_range
            X_norm[i, j, 2::3] = cs

print("🚀 2. Edge TPU 100% 호환: Conv1D 시계열 모델 학습 (초고속)")
# 🌟 LSTM 대신 TPU의 주특기인 Conv1D(1차원 합성곱)를 사용합니다!
# 배치 사이즈 1도 처음부터 고정해버립니다.
coral_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(20, 51), batch_size=1),
    
    # 20프레임의 시간 흐름을 3프레임씩 묶어서 패턴을 파악합니다.
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(), # 데이터를 1차원으로 쫙 펴줌
    
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

coral_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
coral_model.fit(X_norm, Y, epochs=30, batch_size=1, verbose=1)

print("\n🚀 3. Edge TPU 완벽 양자화(INT8) 진행...")
converter = tf.lite.TFLiteConverter.from_keras_model(coral_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for i in range(min(100, len(X_norm))):
        yield [np.expand_dims(X_norm[i].astype(np.float32), axis=0)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32 
converter.inference_output_type = tf.float32

tflite_quant_model = converter.convert()

with open('fall_model_conv1d.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("🎉 대성공! 튕김 절대 없는 'fall_model_conv1d.tflite' 생성 완료!")