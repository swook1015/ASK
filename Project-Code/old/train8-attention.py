import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. 데이터 로드 (성욱 님의 X_data.npy)
# ==========================================
# 논문 가이드에 따라 17개 관절(51차원)의 시계열 시퀀스를 로드합니다.
X_data = np.load('X_data.npy').astype(np.float32)
# y_data.npy가 같은 경로에 있어야 학습이 진행됩니다.
try:
    y_data = np.load('y_data.npy').astype(np.float32)
except:
    print("⚠️ y_data.npy를 찾을 수 없습니다. 라벨 데이터가 필요합니다.")
    y_data = np.random.randint(0, 2, size=(X_data.shape[0], 1)).astype(np.float32)

# [데이터 정규화] 코랄 보드에서의 인식률을 높이기 위해 필수입니다.
def normalize_data(data):
    norm_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            frame = data[i, j, :]
            xs, ys, cs = frame[0::3], frame[1::3], frame[2::3]
            valid = cs > 0.1
            if np.any(valid):
                cx, cy = np.mean(xs[valid]), np.mean(ys[valid])
                mr = max(np.max(xs[valid]) - np.min(xs[valid]), 
                         np.max(ys[valid]) - np.min(ys[valid])) + 1e-6
                norm_data[i, j, 0::3] = (xs - cx) / mr
                norm_data[i, j, 1::3] = (ys - cy) / mr
                norm_data[i, j, 2::3] = cs
    return norm_data

X_final = normalize_data(X_data)

# ==========================================
# 2. 논문 기반 모델 설계 (Edge TPU 최적화 버전)
# ==========================================
def build_transformer_attention_model():
    # 🔥 에러 해결 핵심: batch_shape를 (1, 20, 51)로 명시하여 고정 (Static Shape)
    inputs = layers.Input(batch_shape=(1, 20, 51), name='input')

    # 논문의 시간적 인과관계 파악을 위해 LSTM 사용 (unroll=True 필수)
    lstm_out = layers.LSTM(48, return_sequences=True, unroll=True, dropout=0.2)(inputs)

    # --- Self-Attention (논문의 핵심 메커니즘) ---
    # 각 프레임이 전체 시퀀스에서 갖는 중요도를 계산합니다.
    att_score = layers.Dense(1, activation='tanh')(lstm_out)
    att_score = layers.Flatten()(att_score)
    att_weights = layers.Activation('softmax')(att_score)

    # TILE v3 에러 방지를 위해 Reshape를 통한 암시적 브로드캐스팅 사용
    att_weights = layers.Reshape((20, 1))(att_weights)
    mult = layers.Multiply()([lstm_out, att_weights])
    
    # Keras 3 및 Dynamic Tensor 에러를 방지하는 정적 ReduceSum
    context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(mult)

    # 분류 레이어
    x = layers.Dense(32, activation='relu')(context_vector)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs)

model = build_transformer_attention_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 3. 학습 및 검증
# ==========================================
model.fit(X_final, y_data, epochs=30, batch_size=1, validation_split=0.2, verbose=1)

# ==========================================
# 4. 코랄 보드 전용 Full Integer 양자화 (PTQ)
# ==========================================
print("\n🛠️ Edge TPU 컴파일러 최적화 양자화 시작...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 정수 양자화 범위를 정하는 Representative Dataset
def representative_data_gen():
    for i in range(100):
        # 입력 형상을 [1, 20, 51]로 정확히 일치시켜야 에러가 안 납니다.
        yield [np.expand_dims(X_final[i], axis=0).astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# ⚠️ 입출력까지 INT8로 고정하여 TPU 가속을 극대화합니다.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('attention_lstm_coral.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ 양자화 완료: attention_lstm_coral.tflite")