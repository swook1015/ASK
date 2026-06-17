import os
# 🚨 Keras 2(Legacy) 엔진 강제 사용 (호환성 유지)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

# ==========================================
# ⚙️ 1. 경로 설정
# ==========================================
h5_model_path = './AI/models/lpn_cv_60_legacy_movenet.h5'
tflite_model_path = './AI/models/lpn_cv_60_legacy_movenet_ptq.tflite'
train_data_path = './AI/dataset/CV_Split/X_cv_train.npy'

# ==========================================
# 🧠 2. 캘리브레이션용 데이터 정규화 (학습 때와 100% 동일)
# ==========================================
def load_and_normalize_for_calibration(npy_path, sample_size=500):
    print(f"📦 캘리브레이션 데이터 로드 중... ({npy_path})")
    raw_3d = np.load(npy_path).astype(np.float32)
    
    # 대표 샘플 500개만 사용 (빠른 변환을 위해)
    raw_3d = raw_3d[:sample_size]
    
    N, frames, _ = raw_3d.shape
    raw_3d_reshaped = raw_3d.reshape(N, frames, 13, 3)
    
    # 영점 조절
    root_xyz = raw_3d_reshaped[:, :, 0:1, :] 
    centered_3d = raw_3d_reshaped - root_xyz 
    
    # 스케일 정규화 (-1 ~ 1)
    max_dist = np.max(np.linalg.norm(centered_3d, axis=-1), axis=-1, keepdims=True)
    max_dist = np.expand_dims(max_dist, axis=-1)
    max_dist[max_dist == 0] = 1e-6 
    
    normalized_3d = centered_3d / max_dist
    
    # Z축 제거 (2D 형태)
    normalized_2d = normalized_3d[:, :, :, 0:2] 
    X = normalized_2d.reshape(N, frames, 26) 
    
    return X

calib_data = load_and_normalize_for_calibration(train_data_path)
print(f"✅ 데이터 준비 완료: {calib_data.shape}")

# 제너레이터 함수
def representative_dataset():
    for i in range(len(calib_data)):
        # TFLite 모델 입력에 맞게 (1, 60, 26) 차원으로 확장해서 주입
        yield [np.expand_dims(calib_data[i], axis=0).astype(np.float32)]

# ==========================================
# ⚙️ 3. TFLite 변환 및 INT8 양자화 실행
# ==========================================
print(f"\n🚀 H5 모델 로드 중: {h5_model_path}")
model = tf.keras.models.load_model(h5_model_path)

print("⚙️ TFLite 변환 및 INT8 양자화 시작...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Edge TPU 호환성을 위한 Full Integer Quantization 강제
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# 파일 저장
os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"\n🎉 일반 TFLite (PTQ) 변환 및 저장 완료!")
print(f"📂 저장 위치: {tflite_model_path}")