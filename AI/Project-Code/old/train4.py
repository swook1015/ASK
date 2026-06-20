import numpy as np
import tensorflow as tf

print("🚀 1. 데이터 로드 및 전처리...")
# 정답지(Y) 데이터 이름이 다르다면 수정해 주세요 (예: y_data.npy 또는 Y_data.npy)
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

print("🚀 2. Keras 3 환경에서 새로운 뇌(모델) 초고속 학습 진행...")
# 학습할 때는 자유로운 형태로 학습합니다.
train_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(20, 51)),
    tf.keras.layers.LSTM(48),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

train_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 데이터가 작으니 30번만 빠르게 반복 학습합니다. (10초 컷)
train_model.fit(X_norm, Y, epochs=30, batch_size=32, verbose=1)

print("\n🚀 3. Edge TPU 전용 껍데기(Unrolled) 생성 및 뇌 이식...")
# 🌟 여기가 핵심: 배치 사이즈 1 고정 + unroll=True (루프 박살내기) 🌟
coral_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(20, 51), batch_size=1),
    tf.keras.layers.LSTM(48, unroll=True), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 방금 학습한 똑똑한 뇌(가중치)를 코랄 전용 껍데기에 그대로 복사합니다!
coral_model.set_weights(train_model.get_weights())

print("🚀 4. Edge TPU 완벽 양자화(INT8) 진행...")
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

with open('fall_model_coral_ready.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("🎉 대성공! 완벽한 Edge TPU용 'fall_model_coral_ready.tflite' 생성 완료!")