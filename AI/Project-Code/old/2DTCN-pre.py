import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
print("📦 데이터 불러오는 중...")
X = np.load('./AI/dataset/2D-TCN/ntu_X_60frames_coral.npy').astype(np.float32)
y = np.load('./AI/dataset/2D-TCN/ntu_y_60frames_onehot.npy').astype(np.float32)

# One-hot -> Binary (Sigmoid용)
y_binary = np.argmax(y, axis=1).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
print(f"✅ 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

# 클래스 가중치 계산
normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
total = normal_count + fall_count
class_weights = {0: (1 / normal_count) * (total / 2.0), 1: (1 / fall_count) * (total / 2.0)}

# ==========================================
# 2. 코랄 최적화 Strided Block (Dilation 완벽 대체)
# ==========================================
def strided_residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.25):
    shortcut = x
    
    # 🚨 Dilation을 버리고, Stride를 사용해 시간축을 압축하며 시야를 넓힙니다.
    x = layers.Conv2D(filters, (kernel_size, 1), 
                      strides=(stride, 1), 
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 🚨 메인 경로가 Stride로 줄어들었으면, Shortcut도 똑같이 줄여줍니다.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=(stride, 1), padding='same')(shortcut)

    return layers.Add()([shortcut, x])

# ==========================================
# 3. 모델 구성 (코랄 보드 네이티브)
# ==========================================
def build_coral_strided_net(input_shape=(60, 1, 39)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 프레임 압축 과정: 60 -> 60 -> 30 -> 15 -> 8
    # 엣지 TPU 메모리에 부담이 없으므로 채널을 다시 논문 수준으로 올릴 수 있습니다!
    x = strided_residual_block(x, 64,  stride=1)
    x = strided_residual_block(x, 128, stride=2)
    x = strided_residual_block(x, 256, stride=2)
    x = strided_residual_block(x, 256, stride=2) # 512 대신 256으로 안전장치

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Coral_Strided_Native")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_coral_strided_net()
model.summary()

# ==========================================
# 4. 학습 실행
# ==========================================
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
checkpoint = ModelCheckpoint('./AI/models/tcn_fall_detector_best.h5', monitor='val_loss', save_best_only=True)

print("\n🚀 코랄 네이티브 Strided 모델 학습을 시작합니다!")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1, 
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)