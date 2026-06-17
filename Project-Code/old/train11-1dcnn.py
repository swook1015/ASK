import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 데이터 준비 (데이터 자체를 2D 이미지 형태로 강제 변환)
try:
    X_raw = np.load('X_train.npy').astype(np.float32)
    y_data = np.load('y_train.npy').astype(np.float32)
except FileNotFoundError:
    print("❌ 에러: 데이터 파일이 없습니다.")
    exit()

indices = np.arange(X_raw.shape[0])
np.random.shuffle(indices)
X_raw, y_data = X_raw[indices], y_data[indices]

def normalize_pose(data):
    norm_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            frame = data[i, j, :]
            xs, ys, cs = frame[0::3], frame[1::3], frame[2::3]
            valid = cs > 0.1
            if np.any(valid):
                cx, cy = np.mean(xs[valid]), np.mean(ys[valid])
                mr = max(np.max(xs[valid]) - np.min(xs[valid]), np.max(ys[valid]) - np.min(ys[valid])) + 1e-6
                norm_data[i, j, 0::3] = (xs - cx) / mr
                norm_data[i, j, 1::3] = (ys - cy) / mr
                norm_data[i, j, 2::3] = cs
    return norm_data

X_final = normalize_pose(X_raw)

# 🔥 아예 데이터 자체를 (20, 51, 1)의 2D 형태로 만듭니다. (Reshape 레이어 삭제 위함)
X_final_2d = np.expand_dims(X_final, axis=-1)

# 2. 모델 찍어내는 틀 (배치 사이즈를 변수로 받습니다)
def create_model(batch_size=None):
    # 입력층부터 2D(이미지) 형태를 받습니다.
    inputs = layers.Input(batch_shape=(batch_size, 20, 51, 1), name='input')
    
    x = layers.Conv2D(32, kernel_size=(3, 51), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 1), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

# ==========================================
# 🥇 STEP 1: 학습용 모델 (Dynamic Batch)
# ==========================================
print("🚀 [1단계] 학습용 모델 생성 및 학습 (배치 32)...")
train_model = create_model(batch_size=None) # 학습 땐 자유롭게 놔둠
train_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_model.fit(X_final_2d, y_data, epochs=30, batch_size=32)

# ==========================================
# 🥈 STEP 2: 변환용 모델 (Static Batch) 및 가중치 복사
# ==========================================
print("\n🚀 [2단계] 코랄 보드 전용 정적 모델 생성 (배치 1 고정)...")
tpu_model = create_model(batch_size=1) # 🔥 코랄 보드가 요구하는 절대 규격

# 똑같은 구조이므로 뇌(가중치)만 그대로 이식합니다.
tpu_model.set_weights(train_model.get_weights())
print("✅ 가중치 이식 완료!")

# ==========================================
# 🥉 STEP 3: Edge TPU 양자화
# ==========================================
print("\n🛠️ [3단계] 정적 모델 양자화 시작...")
converter = tf.lite.TFLiteConverter.from_keras_model(tpu_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for i in range(100):
        yield [X_final_2d[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
    with open('the_absolute_final.tflite', 'wb') as f:
        f.write(tflite_model)
    print("\n✅ 변환 완료! 이제 컴파일러를 돌려보세요. Dynamic 에러는 절대 안 뜹니다.")
except Exception as e:
    print(f"\n❌ 변환 실패: {e}")