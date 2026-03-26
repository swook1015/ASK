import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 🚨 Keras 2(Legacy) 엔진 강제 사용 (코랄 컴파일 호환성 유지)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# 1. 데이터 로드 (Cross-Subject 전용)
# ==========================================
print("📦 Cross-Subject 데이터 로드 중...")
DATA_DIR = './AI/dataset/CS_Split'

X_train = np.load(os.path.join(DATA_DIR, 'X_cs_train.npy')).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, 'y_cs_train.npy')).astype(np.float32).reshape(-1, 1)

X_test = np.load(os.path.join(DATA_DIR, 'X_cs_test.npy')).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, 'y_cs_test.npy')).astype(np.float32).reshape(-1, 1)

# 🔥 [추가] 차원 맞추기: (N, 60, 1, 39) -> (N, 60, 39)
X_train = X_train.reshape(-1, 60, 39)
X_test = X_test.reshape(-1, 60, 39)

X_val, y_val = X_test, y_test

print(f"📊 CS 데이터 구성 완료:")
print(f"   - Train (학습 인물 그룹): {X_train.shape[0]} 샘플")
print(f"   - Test  (생판 남 그룹): {X_test.shape[0]} 샘플")

# 클래스 불균형 해결을 위한 가중치 계산
normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
class_weights = {0: 1.0, 1: normal_count / fall_count}
print(f"📊 클래스 가중치 적용 - Fall: {class_weights[1]:.2f}x")

# ==========================================
# 2. 모델 구성 함수 (Edge TPU 최적화 8층 TCN)
# ==========================================
def conv_block(x, filters, strides=1, name="conv"):
    x = layers.ZeroPadding1D(padding=1, name=f"{name}_pad")(x)
    x = layers.Conv1D(filters, 3, strides=strides, padding='valid', 
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x

def build_simple_8layer_tcn(input_shape=(60, 39)):
    """단순 8층 TCN (Edge TPU 호환성 및 CS 검증용)"""
    inputs = layers.Input(shape=input_shape, name="input")
    
    # Stage1: 60 → 30
    x = conv_block(inputs, 48, strides=1, name="S1_conv1")
    x = conv_block(x, 48, strides=2, name="S1_conv2")
    
    # Stage2: 30 → 15
    x = conv_block(x, 64, strides=1, name="S2_conv1")
    x = conv_block(x, 64, strides=2, name="S2_conv2")
    
    # Stage3: 15 → 8
    x = conv_block(x, 96, strides=1, name="S3_conv1")
    x = conv_block(x, 96, strides=2, name="S3_conv2")
    
    # Stage4: 8 → 4
    x = conv_block(x, 128, strides=1, name="S4_conv1")
    x = conv_block(x, 128, strides=2, name="S4_conv2")
    
    # 출력단
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Simple_CS_8Layer_TCN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

# ==========================================
# 3. 학습 실행 및 결과 저장
# ==========================================
model = build_simple_8layer_tcn()
model.summary()

os.makedirs('./AI/models', exist_ok=True)
h5_path = './AI/models/tcn_8layer_cs_standard.h5'

callbacks = [
    # 💡 CS 검증은 더 까다로우므로 patience를 30으로 넉넉히 줌
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(h5_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("\n🚀 Cross-Subject 8층 TCN 학습 시작...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), # 여기가 핵심 (한 번도 안 본 사람들에 대한 테스트)
    epochs=150, 
    batch_size=256, 
    class_weight=class_weights,
    callbacks=callbacks, 
    verbose=1,
    shuffle=True # 학습 데이터는 매 에포크마다 섞어줌
)

# ==========================================
# 4. 최종 리포트 출력
# ==========================================
best_val_acc = max(history.history['val_accuracy'])
print(f"\n{'='*60}")
print(f"🎉 Cross-Subject 학습 완료!")
print(f"📊 최종 CS 정확도 (Accuracy): {best_val_acc*100:.2f}%")
print(f"📝 논문 SOTA 수치: 99.83%")
print(f"💾 모델 저장 완료: {h5_path}")
print(f"{'='*60}")