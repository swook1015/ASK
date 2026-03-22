import os
# Keras 2(Legacy) 엔진 강제 사용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
# ==========================================
# 1. 통합 데이터 로드 및 전처리
# ==========================================
print("📦 통합 데이터(NTU + URFD) 불러오는 중...")
# 병합해서 현재 폴더에 저장한 파일 사용
X = np.load('./combined_train_X.npy').astype(np.float32)
y = np.load('./combined_tarain_y.npy').astype(np.float32)

# 이미 MergeData.py에서 4차원 껍데기도 벗기고 정답지도 0/1로 맞췄으므로, 
# 모델 입력 규격에 맞게 세로형(N, 1)으로만 세워줍니다.
y_binary = y.reshape(-1, 1)

# 학습용(Train)과 검증용(Val)을 8:2로 분리 (stratify로 낙상/정상 비율 유지)
X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
print(f"✅ 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

# 데이터 불균형 해소를 위한 가중치(Class Weights) 계산
normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
total = normal_count + fall_count
class_weights = {0: (1 / normal_count) * (total / 2.0), 1: (1 / fall_count) * (total / 2.0)}
print(f"⚖️ 클래스 가중치 적용: 정상(0)={class_weights[0]:.4f}, 낙상(1)={class_weights[1]:.4f}")

# ==========================================
# 2. 오리지널 논문 규격 TCN Block (Dilation 부활)
# ==========================================
def tcn_residual_block(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.25):
    shortcut = x
    
    # 🚨 [핵심 3] 논문의 핵심인 Dilation 적용! 
    # 프레임을 줄이지 않고(stride=1) 시야를 넓혀 세밀한 프레임 정보를 끝까지 유지합니다.
    x = layers.Conv1D(filters, kernel_size, 
                      padding='same', 
                      dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(filters, kernel_size, 
                      padding='same', 
                      dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 채널 수가 다를 경우 1x1 Conv로 맞춰줌
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)

    return layers.Add()([shortcut, x])

# ==========================================
# 3. 모델 구성 (논문 Dilation + TPU 1D 구조)
# ==========================================
def build_paper_level_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 논문 구조: Dilation rate를 1, 2, 4, 8로 기하급수적으로 증가
    x = tcn_residual_block(x, 64, dilation_rate=1)
    x = tcn_residual_block(x, 64, dilation_rate=2)
    x = tcn_residual_block(x, 128, dilation_rate=4)
    x = tcn_residual_block(x, 128, dilation_rate=8)

    # 🚨 [핵심 4] Flatten 대신 GlobalAveragePooling1D 사용
    # 논문 및 최신 구조의 정석. 파라미터를 크게 줄여 오버피팅을 막고 연산을 가볍게 만듭니다.
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Paper_Level_1D_TCN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_paper_level_tcn()
model.summary()

# ==========================================
# 4. 학습 실행
# ==========================================
# 저장할 폴더가 없으면 에러가 날 수 있으므로 미리 만들어줍니다.
os.makedirs('./AI/models', exist_ok=True)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
checkpoint = ModelCheckpoint('./AI/models/tcn_fall_detector_paper_best.h5', monitor='val_loss', save_best_only=True)

print("\n🚀 논문 규격 1D-TCN 모델 학습을 시작합니다!")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, 
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

print("\n🎉 모든 학습이 완료되었습니다! 최고 성능의 모델이 './AI/models/tcn_fall_detector_paper_best.h5'에 저장되었습니다.")