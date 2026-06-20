import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Reshape, Multiply, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.utils import class_weight

# 1. 데이터 로드 및 셔플
X, Y = np.load('X_data.npy'), np.load('Y_data.npy')
indices = np.arange(X.shape[0]); np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

# 2. 정규화 (유지)
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

# 🌟 [개선 1] 부족한 낙상 데이터를 위해 가중치 계산
weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
class_weights = {0: weights[0], 1: weights[1]}

# 3. 더 깊고 정교한 모델 설계
inputs = Input(shape=(20, 51))
lstm_1 = LSTM(128, return_sequences=True, unroll=True)(inputs) # 유닛 수 증가 (64 -> 128)
lstm_1 = BatchNormalization()(lstm_1)
lstm_2 = LSTM(64, return_sequences=True, unroll=True)(lstm_1) # 레이어 추가
lstm_2 = BatchNormalization()(lstm_2)

# Attention (그대로 유지하되 층을 통과한 데이터 사용)
att_weights = Flatten()(lstm_2)
att_weights = Dense(20, activation='softmax')(att_weights)
att_weights = Reshape((20, 1))(att_weights)
context = Multiply()([lstm_2, att_weights])

context = Flatten()(context)
dense = Dense(64, activation='relu')(context)
dense = Dropout(0.5)(dense) # 드롭아웃 강화 (0.3 -> 0.5)
outputs = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])

# 🌟 [개선 2] 똑똑한 학습 제어 (콜백함수)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# 4. 학습 시작
print("🔥 하이퍼 퍼포먼스 학습 모드 가동...")
model.fit(X_norm, Y, epochs=100, batch_size=32, validation_split=0.2, 
          class_weight=class_weights, # 가중치 적용
          callbacks=[early_stop, lr_reducer])

# 5. TFLite 변환
model.export("hyper_model")
converter = tf.lite.TFLiteConverter.from_saved_model("hyper_model")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('fall_model_coral_pro.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ 최종 모델 생성 완료: fall_model_coral_pro.tflite")