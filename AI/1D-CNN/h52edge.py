import tensorflow as tf
import numpy as np
import os

# 1. 경로 설정
model_path = './AI/models/lpn_paper_3d_lifter.h5'
rep_data_path = './AI/dataset/LPN-train/ntu_lpn_input_2d_rep.npy'
tflite_save_path = './AI/models/lpn_3d_lifter_quant.tflite'

# 2. 대표 데이터셋 로드 (양자화 영점 조절용)
# LPN의 입력 규격인 (None, 26)에 맞춰야 합니다.
rep_data = np.load(rep_data_path).astype(np.float32)

def representative_dataset_gen():
    # 대표 데이터 500개를 하나씩 모델에 흘려보내며 범위를 계산합니다.
    for i in range(len(rep_data)):
        input_value = np.expand_dims(rep_data[i], axis=0)
        yield [input_value]

# 3. TFLite 변환 시작
print("🚀 TFLite Full Integer 양자화 변환 시작...")
model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- [핵심 양자화 설정] ---
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# 코랄 보드(Edge TPU) 호환을 위해 필수 설정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# -------------------------

tflite_model = converter.convert()

# 4. 저장
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ 변환 완료! 저장 위치: {tflite_save_path}")
print("이제 이 파일을 'edgetpu_compiler'로 컴파일하면 코랄 보드 준비 끝입니다!")