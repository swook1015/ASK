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



# 🔥 [핵심 추가] 학습 데이터 셔플 (짝을 맞춰서 랜덤하게 섞기)

print("🔀 학습 데이터 셔플 중...")

indices = np.arange(X_train_raw.shape[0])

np.random.shuffle(indices)

X_train_raw = X_train_raw[indices]

y_train = y_train[indices]



# 🔥 전처리 삭제: 원본 데이터를 그대로 4D 형태로 변환(확장)만 해줍니다.

print("🧹 원본(Raw) 데이터 4D 변환 중... (정규화 제거)")

X_train = np.expand_dims(X_train_raw, axis=-1)

X_test = np.expand_dims(X_test_raw, axis=-1)



# 2. 모델 틀 정의 (기존 51개 입력 구조 그대로 유지)

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

print("\n🚀 [1단계] 실전용(Raw) 모델 생성 및 학습...")

train_model = create_model(batch_size=None)

train_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# epochs=30으로 충분히 학습합니다.

train_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))



# ==========================================

# 📊 STEP 2: 학술대회용 성적표 출력

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

with open('raw_model.tflite', 'wb') as f:

    f.write(tflite_model)

print("\n✅ 모든 과정 완료! 'raw_model.tflite' 생성됨.")