import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. 세션 클리어 (이전의 지저분한 상태를 모두 삭제)
tf.keras.backend.clear_session()

# 데이터 로드 (X_data.npy, y_data.npy가 있다고 가정)
X_final = np.load('X_data.npy').astype(np.float32)
# [성욱 님의 정규화 함수 통과 후 데이터라고 가정]
y_data = np.load('y_data.npy').astype(np.float32)

# 2. 모델 설계 (불필요한 Dropout, Seed 요소를 완전히 제거)
def build_clean_inference_model():
    # 고정 배치 1
    inputs = layers.Input(batch_shape=(1, 20, 51), name='input')

    # 🔥 [수정] dropout=0.2를 아예 삭제합니다. (추론 시엔 필요 없음)
    lstm_out = layers.LSTM(48, return_sequences=True, unroll=True)(inputs)

    # --- 🌟 정적 Attention 구조 ---
    att_score = layers.Dense(1, activation='tanh')(lstm_out)
    
    # 3차원 Softmax가 에러를 낼 수 있으므로 2차원으로 펴서 확실히 처리
    att_score_flat = layers.Reshape((20,))(att_score)
    att_weights_flat = layers.Softmax(name='attention_weights')(att_score_flat)
    att_weights = layers.Reshape((20, 1))(att_weights_flat)

    # 가중합 (Multiply 연산은 정적일 때 매우 안정적임)
    mult = layers.Multiply()([lstm_out, att_weights])
    
    # GlobalAveragePooling1D 대신 직접적인 Mean 연산과 동일한 효과를 내는 레이어 사용
    context_vector = layers.GlobalAveragePooling1D()(mult)

    # 출력층
    x = layers.Dense(32, activation='relu')(context_vector)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs)

model = build_clean_inference_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. 학습
model.fit(X_final, y_data, epochs=1, batch_size=1, verbose=1)

# 4. 양자화 및 변환
print("\n🛠️ [CLEAN] Edge TPU 최적화 변환...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for i in range(100):
        # 정확히 [1, 20, 51] 형상 유지
        yield [np.expand_dims(X_final[i], axis=0).astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()
with open('attention_lstm_clean.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ '깨끗한' 모델 생성 완료! 이제 컴파일러를 돌려보세요.")