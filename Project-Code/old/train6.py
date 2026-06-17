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

print("🚀 2. '어텐션(Attention) LSTM' 모델 구조 생성 및 학습...")

# 입력층 (배치사이즈 1 고정)
inputs = tf.keras.Input(batch_shape=(1, 20, 51))

# LSTM 층 (unroll=True 적용, 어텐션을 위해 모든 시점의 출력 반환)
lstm_out = tf.keras.layers.LSTM(48, return_sequences=True, unroll=True)(inputs)

# 🌟 어텐션(Attention) 메커니즘 🌟
# 각 프레임(시간)이 얼마나 중요한지 점수를 매깁니다.
attention_scores = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
attention_scores = tf.keras.layers.Flatten()(attention_scores)
attention_weights = tf.keras.layers.Activation('softmax', name='attention_weights')(attention_scores)

# 점수를 LSTM 출력에 곱해서 중요한 프레임을 강조합니다.
attention_weights = tf.keras.layers.RepeatVector(48)(attention_weights)
attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
context_vector = tf.keras.layers.Multiply()([lstm_out, attention_weights])
context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)

# 출력층
x = tf.keras.layers.Dense(32, activation='relu')(context_vector)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# 모델 조립 및 학습
attention_model = tf.keras.Model(inputs=inputs, outputs=outputs)
attention_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
attention_model.fit(X_norm, Y, epochs=30, batch_size=1, verbose=1)

print("\n🚀 3. Edge TPU 양자화 진행...")
converter = tf.lite.TFLiteConverter.from_keras_model(attention_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for i in range(min(100, len(X_norm))):
        yield [np.expand_dims(X_norm[i].astype(np.float32), axis=0)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32 
converter.inference_output_type = tf.float32

tflite_quant_model = converter.convert()

with open('fall_model_attention_lstm.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("🎉 '어텐션 LSTM' 양자화 완료 (fall_model_attention_lstm.tflite)")