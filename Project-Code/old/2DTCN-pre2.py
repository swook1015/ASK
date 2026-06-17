import os

# 🚨 [핵심 1] Keras 3의 방해를 차단하고 Keras 2(Legacy) 엔진 강제 사용

os.environ["TF_USE_LEGACY_KERAS"] = "1"



import tf_keras as keras

from tf_keras import layers, models

from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import numpy as np



# ==========================================

# 1. 데이터 로드 및 전처리 (3D 구조로 복원)

# ==========================================

print("📦 데이터 불러오는 중...")

X = np.load('./AI/dataset/2D-TCN/ntu_X_60frames_coral.npy').astype(np.float32)

y = np.load('./AI/dataset/2D-TCN/ntu_y_60frames_onehot.npy').astype(np.float32)



# 🚨 [핵심 2] 불필요한 '가로 1픽셀' 껍데기를 벗겨내고 순수 1D(시계열) 형태로 복원

if len(X.shape) == 4:

    X = np.squeeze(X, axis=2) # (N, 60, 1, 39) -> (N, 60, 39)



y_binary = np.argmax(y, axis=1).reshape(-1, 1)



X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

print(f"✅ 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")



normal_count = np.sum(y_train == 0)

fall_count = np.sum(y_train == 1)

total = normal_count + fall_count

class_weights = {0: (1 / normal_count) * (total / 2.0), 1: (1 / fall_count) * (total / 2.0)}



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

    # 60프레임을 압축하지 않고 시야(Receptive Field)만 61프레임까지 확장하여 전체를 한눈에 파악합니다.

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

