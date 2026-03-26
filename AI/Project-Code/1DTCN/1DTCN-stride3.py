import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 🚨 Keras 2(Legacy) 엔진 강제 사용 (코랄 컴파일 호환성)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# 1. 데이터 로드 및 검증 데이터 분리 저장
# ==========================================
print("📦 데이터 로드 중...")
X = np.load('./combined_train_X.npy').astype(np.float32)
y = np.load('./combined_train_y.npy').astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

val_save_dir = './AI/val_data'
os.makedirs(val_save_dir, exist_ok=True)
np.save(f'{val_save_dir}/val_X.npy', X_val)
np.save(f'{val_save_dir}/val_y.npy', y_val)
print(f"💾 검증 데이터 저장 완료: {val_save_dir}")

normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
class_weights = {0: 1.0, 1: normal_count / fall_count}
print(f"📊 클래스 분포 - Normal: {int(normal_count)}, Fall: {int(fall_count)}")
print(f"📊 클래스 가중치 - Fall: {class_weights[1]:.2f}x")

# ==========================================
# 2. 모델 구성 함수 (개선된 8층)
# ==========================================
def conv_block(x, filters, strides=1, name="conv"):
    """기본 Conv1D 블록 (ZeroPadding + Conv + BN + ReLU)"""
    x = layers.ZeroPadding1D(padding=1, name=f"{name}_pad")(x)
    x = layers.Conv1D(filters, 3, strides=strides, padding='valid', 
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x

def residual_stage(x, filters, stage_name, downsample=True):
    """
    개선된 스테이지: Residual Connection 추가
    - 특징 추출 + 다운샘플링을 하나의 스테이지로
    - Skip connection으로 그래디언트 흐름 개선
    """
    shortcut = x
    
    # Main path
    x = conv_block(x, filters, strides=1, name=f"{stage_name}_conv1")
    
    if downsample:
        x = conv_block(x, filters, strides=2, name=f"{stage_name}_conv2")
        # Shortcut도 다운샘플링 (1x1 conv로 차원 맞춤)
        shortcut = layers.Conv1D(filters, 1, strides=2, padding='valid',
                                  use_bias=False, name=f"{stage_name}_shortcut")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{stage_name}_shortcut_bn")(shortcut)
    else:
        x = conv_block(x, filters, strides=1, name=f"{stage_name}_conv2")
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='valid',
                                      use_bias=False, name=f"{stage_name}_shortcut")(shortcut)
            shortcut = layers.BatchNormalization(name=f"{stage_name}_shortcut_bn")(shortcut)
    
    # Residual connection
    x = layers.Add(name=f"{stage_name}_add")([x, shortcut])
    x = layers.Activation('relu', name=f"{stage_name}_out")(x)
    return x


def build_improved_8layer_tcn(input_shape=(60, 39)):
    """
    개선된 8층 TCN
    
    개선점:
    1. 점진적 필터 증가: 48 → 64 → 128 → 192 (더 부드러운 스케일링)
    2. Residual connections: 양자화 시 그래디언트 안정성
    3. Dropout 추가: 과적합 방지
    
    구조:
    - Stage1: 60f → 30f (48 filters)
    - Stage2: 30f → 15f (64 filters)  
    - Stage3: 15f → 8f  (128 filters)
    - Stage4: 8f → 4f   (192 filters)
    """
    inputs = layers.Input(shape=input_shape, name="input")
    
    # 🔥 개선 1: 점진적 필터 증가 (급격한 점프 방지)
    x = residual_stage(inputs, filters=48,  stage_name="Stage1", downsample=True)  # 60→30
    x = residual_stage(x,      filters=64,  stage_name="Stage2", downsample=True)  # 30→15
    x = residual_stage(x,      filters=128, stage_name="Stage3", downsample=True)  # 15→8
    x = residual_stage(x,      filters=192, stage_name="Stage4", downsample=True)  # 8→4
    
    # 🔥 개선 2: Dropout 추가 (Edge TPU에서도 호환)
    x = layers.Dropout(0.3, name="dropout")(x)
    
    # 출력단
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Improved_8Layer_TCN")
    
    # 🔥 개선 3: Label Smoothing으로 과신 방지
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


def build_simple_8layer_tcn(input_shape=(60, 39)):
    """
    단순 8층 TCN (Residual 없이, Edge TPU 호환성 최대화)
    
    기존 코드와 동일한 구조지만 필터 수 최적화
    """
    inputs = layers.Input(shape=input_shape, name="input")
    
    # Stage1: 60 → 30 (필터 48로 시작 - 더 가벼움)
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
    
    # 출력
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Simple_8Layer_TCN")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


# ==========================================
# 3. 모델 선택 및 빌드
# ==========================================
# 🔥 선택 1: Residual 버전 (성능 우선)
# model = build_improved_8layer_tcn()

# 🔥 선택 2: 단순 버전 (Edge TPU 호환성 우선) - 권장
model = build_simple_8layer_tcn()

model.summary()

# 파라미터 수 출력
total_params = model.count_params()
print(f"\n📐 총 파라미터: {total_params:,} ({total_params/1000:.1f}K)")

# ==========================================
# 4. 학습 실행
# ==========================================
os.makedirs('./AI/models', exist_ok=True)
h5_path = './AI/models/tcn_8layer_improved.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint(h5_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("\n🚀 개선된 8층 TCN 학습 시작...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=256, 
    class_weight=class_weights,
    callbacks=callbacks, 
    verbose=1
)

# ==========================================
# 5. 학습 결과 요약
# ==========================================
best_val_loss = min(history.history['val_loss'])
best_val_acc = max(history.history['val_accuracy'])
print(f"\n{'='*50}")
print(f"🎉 학습 완료!")
print(f"📊 Best Val Loss: {best_val_loss:.4f}")
print(f"📊 Best Val Accuracy: {best_val_acc:.4f}")
print(f"💾 모델 저장: {h5_path}")
print(f"{'='*50}")