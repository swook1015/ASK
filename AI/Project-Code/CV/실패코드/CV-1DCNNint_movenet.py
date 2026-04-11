import os
# 🚨 Keras 2(Legacy) 엔진 강제 사용 (호환성 유지)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# ==========================================
# ⚙️ 1. 경로 설정
# ==========================================
h5_model_path = './AI/models/tcn_8layer_combined_final_movenet.h5'
tflite_model_path = './AI/models/tcn_manual_fixed_movenet.tflite'
train_data_path = './AI/dataset/CV_Split/X_cv_train.npy'

# ==========================================
# 🧠 2. 캘리브레이션 데이터 정규화 (-1 ~ 1 스케일링)
# ==========================================
def load_and_normalize_for_tcn_calibration(npy_path, sample_size=500):
    print(f"📦 TCN 캘리브레이션 데이터 로드 중... ({npy_path})")
    raw_data = np.load(npy_path).astype(np.float32)[:sample_size]
    
    X_reshaped = raw_data.reshape(-1, 60, 13, 3)
    root_xyz = X_reshaped[:, :, 0:1, :] 
    X_centered = X_reshaped - root_xyz 
    
    max_dist = np.max(np.linalg.norm(X_centered, axis=-1), axis=-1, keepdims=True)
    max_dist = np.expand_dims(max_dist, axis=-1)
    max_dist[max_dist == 0] = 1e-6 
    
    X_normalized = X_centered / max_dist
    return X_normalized.reshape(-1, 60, 39)

X_val = load_and_normalize_for_tcn_calibration(train_data_path, 500)
print(f"✅ 데이터 준비 완료: {X_val.shape}")

def rep_gen():
    for i in range(len(X_val)):
        yield [X_val[i:i+1]]

# ==========================================
# 🔥 3. 그래프 동결 및 TFLite 양자화 (Conv1D 강제 호환 트릭)
# ==========================================
print(f"\n🚀 H5 모델 로드 중: {h5_model_path}")
model = tf.keras.models.load_model(h5_model_path)

print("❄️ 모델 그래프 동결 (Concrete Function) 진행 중...")
full_model = tf.function(lambda x: model(x))
concrete_func = full_model.get_concrete_function(tf.TensorSpec([1, 60, 39], tf.float32))
frozen_func = convert_variables_to_constants_v2(concrete_func)

print("⚙️ 동결된 그래프 기반 TFLite INT8 양자화 시작...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)
with open(tflite_model_path, 'wb') as f: 
    f.write(tflite_model)

print(f"\n🎉 최종 TCN 모델 생성 완료!")
print(f"📂 저장 위치: {tflite_model_path}")
print("💡 이제 이 파일을 Colab에 올려서 'edgetpu_compiler -s -d 파일명' 으로 컴파일하세요!")