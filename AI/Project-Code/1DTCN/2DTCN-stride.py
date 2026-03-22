import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 🚨 [핵심 1] Keras 2(Legacy) 엔진 강제 사용 (코랄 컴파일 호환성)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
print("📦 통합 데이터(NTU + URFD) 불러오는 중...")
# 데이터 경로는 오빠 환경에 맞게 확인해줘! 
X = np.load('./combined_train_X.npy').astype(np.float32)
y = np.load('./combined_train_y.npy').astype(np.float32)

y_binary = y.reshape(-1, 1)

# 학습용/검증용 8:2 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"✅ 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

# 클래스 불균형 해소를 위한 가중치 계산 
normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
total = normal_count + fall_count
class_weights = {
    0: (1 / normal_count) * (total / 2.0), 
    1: (1 / fall_count) * (total / 2.0)
}
print(f"⚖️ 클래스 가중치 적용: 정상(0)={class_weights[0]:.4f}, 낙상(1)={class_weights[1]:.4f}")

# ==========================================
# 2. Edge TPU 전용 Stride 기반 모델 설계
# ==========================================
def build_coral_friendly_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape)
    
    # [Block 1] 초기 진입 (60 frames)
    x = layers.Conv1D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # [Block 2] Stride 2로 압축 (60 -> 30)
    x = layers.Conv1D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # [Block 3] Stride 2로 압축 (30 -> 15)
    x = layers.Conv1D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # [Block 4] Stride 2로 압축 (15 -> 8)
    x = layers.Conv1D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # [Output] 전역 평균 풀링 후 출력
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Coral_Optimized_1D_CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_coral_friendly_tcn()
model.summary()

# ==========================================
# 3. 학습 실행 및 저장
# ==========================================
os.makedirs('./AI/models', exist_ok=True)

# 콜백 설정: 성능 안 오르면 자동 중단 및 최적 모델 저장 
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
checkpoint = ModelCheckpoint(
    './AI/models/tcn_fall_detector_stride_best.h5', 
    monitor='val_loss', 
    save_best_only=True
)

print("\n🚀 실시간 Edge TPU 최적화 모델 학습 시작!")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

print("\n🎉 학습 완료! 모델 저장 위치: './AI/models/tcn_fall_detector_stride_best.h5'")