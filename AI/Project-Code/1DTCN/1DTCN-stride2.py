import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 및 전처리
print("📦 데이터 로드 중...")
X = np.load('./combined_train_X.npy').astype(np.float32)
y = np.load('./combined_train_y.npy').astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 클래스 가중치 계산
normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
class_weights = {0: 1.0, 1: normal_count / fall_count}

# 2. 수동 패딩 기반 Edge TPU 최적화 모델 설계
def build_manual_padding_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape)
    
    # Block 1: 60프레임 유지 (수동 패딩)
    x = layers.ZeroPadding1D(padding=1)(inputs) # 양옆 1개씩 패딩
    x = layers.Conv1D(64, 3, padding='valid')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 2: Stride 2 압축 (30프레임으로 하강)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(64, 3, strides=2, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 3: Stride 2 압축 (15프레임으로 하강)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(128, 3, strides=2, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Block 4: Stride 2 압축 (8프레임으로 하강)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(256, 3, strides=2, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Manual_Padding_TCN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_manual_padding_tcn()
model.summary()

# 3. 학습 실행
os.makedirs('./AI/models', exist_ok=True)
h5_path = './AI/models/tcn_manual_padding_best.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ModelCheckpoint(h5_path, monitor='val_loss', save_best_only=True)
]

print("\n🚀 수동 패딩 TCN 학습 시작...")
model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=100, batch_size=256, class_weight=class_weights,
    callbacks=callbacks, verbose=1
)
print(f"🎉 학습 완료! 모델 저장: {h5_path}")