import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Reshape, Multiply, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# 1. 데이터 로드 및 셔플
X, Y = np.load('X_data.npy'), np.load('Y_data.npy')
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

# 2. 정규화 (학습/실행 동일 로직)
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

# 3. 모델 설계 (과적합 방지 특화)
inputs = Input(shape=(20, 51))

# 🌟 데이터 암기를 방지하는 노이즈 레이어
x = GaussianNoise(0.02)(inputs) 

# 🌟 LSTM 유닛을 줄여 모델을 더 가볍게(융통성 있게) 만듦
lstm_out = LSTM(48, return_sequences=True, unroll=True, kernel_regularizer=l2(0.001))(x)

# Attention 레이어
att_weights = Flatten()(lstm_out)
att_weights = Dense(20, activation='softmax')(att_weights)
att_weights = Reshape((20, 1))(att_weights)
context = Multiply()([lstm_out, att_weights])

# 판단부
context = Flatten()(context)
dense = Dense(32, activation='relu')(context)
# 🌟 드롭아웃을 0.6으로 높여 특정 상황에 집착하지 않게 함
dense = Dropout(0.6)(dense) 
outputs = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# 4. 학습 (EarlyStopping으로 최고 시점에 멈춤)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model.fit(X_norm, Y, epochs=40, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# 5. TFLite 변환
model.export("final_model_folder")
converter = tf.lite.TFLiteConverter.from_saved_model("final_model_folder")
tflite_model = converter.convert()
with open('fall_model_final.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ 최종 모델 생성 완료!")