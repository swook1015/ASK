import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Reshape, Multiply, BatchNormalization
from tensorflow.keras.models import Model

# 1. 데이터 로드
print("📂 데이터 로딩 중...")
X = np.load('X_data.npy')
Y = np.load('Y_data.npy')

# 🌟 [해결책 1] 데이터 셔플 (Shuffle) - 이게 없으면 학습이 안 됩니다!
print("🔀 데이터를 무작위로 섞는 중...")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# 2. 상대좌표 정규화 (Robust 로직)
print("🛠️ 좌표 정규화 진행 중...")
X_norm = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        frame = X[i, j, :]
        xs, ys, cs = frame[0::3], frame[1::3], frame[2::3]
        valid = cs > 0.1
        if np.any(valid):
            cx, cy = np.mean(xs[valid]), np.mean(ys[valid])
            max_range = max(np.max(xs[valid]) - np.min(xs[valid]), 
                            np.max(ys[valid]) - np.min(ys[valid])) + 1e-6
            X_norm[i, j, 0::3] = (xs - cx) / max_range
            X_norm[i, j, 1::3] = (ys - cy) / max_range
            X_norm[i, j, 2::3] = cs

# 3. 모델 설계 (코랄 보드 최적화 + 성능 보강)
inputs = Input(shape=(20, 51))
lstm_out = LSTM(64, return_sequences=True, unroll=True)(inputs)
lstm_out = BatchNormalization()(lstm_out) # 학습 안정화

# Attention
att_weights = Flatten()(lstm_out)
att_weights = Dense(20, activation='softmax')(att_weights)
att_weights = Reshape((20, 1))(att_weights)
context = Multiply()([lstm_out, att_weights])

# 판단부
context = Flatten()(context)
dense = Dense(64, activation='relu')(context)
dense = Dropout(0.3)(dense)
outputs = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), 
              loss='binary_crossentropy', metrics=['accuracy'])

# 4. 학습
print("🔥 학습 시작... 이제 정확도가 제대로 올라갈 겁니다.")
model.fit(X_norm, Y, epochs=50, batch_size=32, validation_split=0.2)

# 5. 🌟 [해결책 2] TFLite 변환 (Keras 3 전용 export 방식)
print("💾 모델 내보내기 및 TFLite 변환 중...")
model.export("temp_export_model") # model.save 대신 export 사용

converter = tf.lite.TFLiteConverter.from_saved_model("temp_export_model")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('fall_model_coral.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ 성공! 'fall_model_coral.tflite'가 생성되었습니다.")