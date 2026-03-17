import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ⚙️ 설정
X_urfd = np.load('X_urfd.npy').astype(np.float32) # (샘플수, 20, 26)
y_urfd = np.load('y_urfd.npy').astype(np.float32)
X_urfd = np.expand_dims(X_urfd, axis=-1)



print("\n🚀 전이 학습 시작...")
# 1. NTU 모델 로드 및 특징 추출기 동결
model = tf.keras.models.load_model('ntu_pretrained_base.h5')
for layer in model.layers:
    if 'conv' in layer.name or 'bn' in layer.name:
        layer.trainable = False # NTU 지식 보호

# 2. URFD 파인튜닝
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_urfd, y_urfd, epochs=20, batch_size=16, validation_split=0.2)

# 3. Edge TPU용 정적 모델로 변환 및 양자화
def create_static_model():
    inputs = layers.Input(batch_shape=(1, 20, 26, 1))
    outputs = model(inputs)
    return models.Model(inputs=inputs, outputs=outputs)

static_model = create_static_model()
converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_data_gen():
    for i in range(100):
        yield [X_urfd[i:i+1].astype(np.float32)]



converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open('final_fall_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("\n✅ 최종 TFLite 모델 생성 완료!")