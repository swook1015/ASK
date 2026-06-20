import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Activation, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

print("📂 데이터 불러오는 중...")
X = np.load('lstm/X_data.npy')
Y = np.load('lstm/Y_data.npy')

# ========================================================
# 🌟 [핵심] 데이터 정규화 (Location & Scale Invariance)
# 화면 구석에서 넘어지든 중앙에서 넘어지든 똑같이 인식하게 만듭니다.
# ========================================================
print("🛠️ 좌표 데이터를 AI가 좋아하는 형태로 가공 중...")
X_norm = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        frame = X[i, j, :]
        xs = frame[0::3]
        ys = frame[1::3]
        cs = frame[2::3]
        
        # 신뢰도가 0.1 이상인 '보이는 관절'만 추려냄
        valid_idx = cs > 0.1
        if np.any(valid_idx):
            # 1. 사람의 중심점 찾기
            cx = np.mean(xs[valid_idx])
            cy = np.mean(ys[valid_idx])
            
            # 2. 사람의 크기(Bounding Box) 구하기
            w = np.max(xs[valid_idx]) - np.min(xs[valid_idx])
            h = np.max(ys[valid_idx]) - np.min(ys[valid_idx])
            scale = max(w, h) + 1e-6 # 0으로 나누는 것 방지
            
            # 3. 중심을 (0,0)으로 옮기고, 크기를 -1 ~ 1 사이로 압축!
            X_norm[i, j, 0::3] = np.where(cs > 0.1, (xs - cx) / scale, 0)
            X_norm[i, j, 1::3] = np.where(cs > 0.1, (ys - cy) / scale, 0)
            X_norm[i, j, 2::3] = cs # 신뢰도 점수는 그대로 유지

# ========================================================
# 🧠 어텐션 LSTM 모델 설계 (가벼우면서도 강력한 구조)
# ========================================================
inputs = Input(shape=(20, 51))

# LSTM 층
lstm_out = LSTM(64, return_sequences=True)(inputs)

# Attention 메커니즘 (가장 중요한 동작 프레임에 점수 몰아주기)
attention_scores = Dense(1, activation='tanh')(lstm_out)
attention_weights = Activation('softmax')(attention_scores)
context_vector = Multiply()([lstm_out, attention_weights])

# 시간에 따른 데이터 압축
context_vector = Lambda(lambda x: K.sum(x, axis=1))(context_vector)

# 판단 층
dense = Dense(32, activation='relu')(context_vector)
dense = Dropout(0.3)(dense)
outputs = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🔥 어텐션 LSTM 학습 시작...")
model.fit(X_norm, Y, epochs=30, batch_size=32, validation_split=0.2)

# ========================================================
# 📦 코랄 보드용 TFLite 파일로 변환 및 저장
# ========================================================
print("💾 코랄 보드용 모델로 변환 중...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True
tflite_model = converter.convert()

with open('fall_model_robust.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ 성공! 'fall_model_robust.tflite' 파일이 생성되었습니다.")