import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# 1. 데이터 로드 (Train + Test 둘 다 불러옵니다)
try:
    X_train_raw = np.load('X_train.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    X_test_raw = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    print(f"✅ 데이터 로드 완료: 학습용({len(X_train_raw)}), 테스트용({len(X_test_raw)})")
except FileNotFoundError:
    print("❌ 에러: .npy 파일이 없습니다.")
    exit()

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

print("🧹 데이터 정규화 및 4D 변환 중...")
X_train = np.expand_dims(normalize_pose(X_train_raw), axis=-1)
X_test = np.expand_dims(normalize_pose(X_test_raw), axis=-1)

# 2. 모델 틀 정의
def create_model(batch_size=None):
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
# 🥇 STEP 1: 학습 (Dynamic Batch)
# ==========================================
print("\n🚀 [1단계] 학습용 모델 생성 및 학습...")
train_model = create_model(batch_size=None)
train_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# epochs=30으로 충분히 학습합니다.
train_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# ==========================================
# 📊 STEP 2: 학술대회용 성적표 출력 (중요!)
# ==========================================
print("\n📊 [2단계] 테스트 데이터(Unseen) 성능 평가...")
y_pred_prob = train_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n" + "="*40)
print("📝 논문에 적을 최종 성능 지표 (Test Set)")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))

# 민감도(Sensitivity) 계산
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
print(f"🔥 최종 민감도(Recall): {sensitivity*100:.2f}%")
print("="*40)

# ==========================================
# 🥈 STEP 3: TPU 변환 (Static Batch 1)
# ==========================================
print("\n🚀 [3단계] 코랄 보드용 정적 모델 생성 및 가중치 이식...")
tpu_model = create_model(batch_size=1)
tpu_model.set_weights(train_model.get_weights())

# 🥉 STEP 4: 양자화 및 변환
print("🛠️ [4단계] 양자화 시작...")
converter = tf.lite.TFLiteConverter.from_keras_model(tpu_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def rep_data_gen():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open('the_absolute_final.tflite', 'wb') as f:
    f.write(tflite_model)
print("\n✅ 모든 과정 완료! 'raw-edgetpu.tflite' 생성됨.")