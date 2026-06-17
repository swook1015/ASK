import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import shuffle

# 🚨 Keras 2(Legacy) 엔진 강제 사용 (코랄 컴파일 호환성 유지)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# 🔥 영점 조절 및 스케일 정규화 (Normalization) 함수
# ==========================================
def normalize_poses(X, do_center=True):
    """(N, 60, 39) 형태의 3D 포즈 데이터를 골반 기준으로 정규화(-1 ~ 1)"""
    X_reshaped = X.reshape(-1, 60, 13, 3)
    
    # 1. 영점 조절 (필요한 경우만 적용)
    if do_center:
        root_xyz = X_reshaped[:, :, 0:1, :]  # 골반 좌표 추출
        X_centered = X_reshaped - root_xyz   # 골반을 0,0,0으로
    else:
        X_centered = X_reshaped              # 이미 센터링된 데이터

    # 2. 🎯 스케일 정규화 (LPN 출력 스케일과 완벽 동기화)
    max_dist = np.max(np.linalg.norm(X_centered, axis=-1), axis=-1, keepdims=True)
    max_dist = np.expand_dims(max_dist, axis=-1)
    max_dist[max_dist == 0] = 1e-6 # 0으로 나누기 방지
    
    X_normalized = X_centered / max_dist
    
    return X_normalized.reshape(-1, 60, 39)

# ==========================================
# 1. 데이터 로드 및 정합성 수정 (라벨 & 센터링 & 스케일링)
# ==========================================
print("📦 데이터 로드 및 파이프라인 정합성 검토 시작...")

# --- (1) NTU CV 데이터 로드 ---
NTU_DIR = './AI/dataset/CV_Split'
X_ntu_train = np.load(os.path.join(NTU_DIR, 'X_cv_train.npy')).astype(np.float32)
y_ntu_train = np.load(os.path.join(NTU_DIR, 'y_cv_train.npy')).astype(np.float32).reshape(-1, 1)

X_test = np.load(os.path.join(NTU_DIR, 'X_cv_test.npy')).astype(np.float32)
y_test = np.load(os.path.join(NTU_DIR, 'y_cv_test.npy')).astype(np.float32).reshape(-1, 1)

# 🎯 NTU 데이터 영점 조절 및 스케일 정규화 동시 적용
print("🎯 NTU 데이터(Train/Test) 영점 조절 및 스케일 정규화 적용 중...")
X_ntu_train = normalize_poses(X_ntu_train, do_center=True)
X_test = normalize_poses(X_test, do_center=True)

# --- (2) URFD 데이터 로드 ---
URFD_DIR = 'C:/Users/pcroom2/Desktop/ASK/AI/dataset/URFD_3D_NPY'
X_urfd_train = np.load(os.path.join(URFD_DIR, 'urfd_3d_train_X.npy')).astype(np.float32)
y_urfd_train = np.load(os.path.join(URFD_DIR, 'urfd_3d_train_y.npy')).astype(np.float32).reshape(-1, 1)

X_val = np.load(os.path.join(URFD_DIR, 'urfd_3d_vaild_X.npy')).astype(np.float32)
y_val = np.load(os.path.join(URFD_DIR, 'urfd_3d_vaild_y.npy')).astype(np.float32).reshape(-1, 1)

# 🎯 URFD는 이미 센터링 되었으므로 스케일 정규화만 적용!
print("🎯 URFD 데이터(Train/Val) 스케일 정규화 적용 중...")
X_urfd_train = normalize_poses(X_urfd_train, do_center=False)
X_val = normalize_poses(X_val, do_center=False)

# --- (3) URFD Train 데이터 20배 오버샘플링 ---
X_urfd_boost = np.repeat(X_urfd_train, 20, axis=0)
y_urfd_boost = np.repeat(y_urfd_train, 20, axis=0)

# --- (4) 최종 학습셋 병합 및 셔플 ---
X_train = np.concatenate([X_ntu_train, X_urfd_boost], axis=0)
y_train = np.concatenate([y_ntu_train, y_urfd_boost], axis=0)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(f"📊 최종 데이터 구성 확인:")
print(f"   - Train (NTU+URFD 병합): {X_train.shape[0]} 샘플")
print(f"   - Valid (URFD 전용):     {X_val.shape[0]} 샘플")
print(f"   - Test  (NTU CV 전용):    {X_test.shape[0]} 샘플")

# 클래스 가중치 계산 (정확한 0, 1 라벨 기반)
normal_count = np.sum(y_train == 0)
fall_count = np.sum(y_train == 1)
class_weights = {0: 1.0, 1: normal_count / fall_count}
print(f"⚖️ 낙상 가중치: {class_weights[1]:.2f}x")

# ==========================================
# 2. 모델 구성 (Edge TPU 최적화 8층 TCN)
# ==========================================
def conv_block(x, filters, strides=1, name="conv"):
    x = layers.ZeroPadding1D(padding=1, name=f"{name}_pad")(x)
    x = layers.Conv1D(filters, 3, strides=strides, padding='valid', 
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x

def build_8layer_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape, name="input")
    x = conv_block(inputs, 48, strides=1, name="S1_conv1")
    x = conv_block(x, 48, strides=2, name="S1_conv2")
    x = conv_block(x, 64, strides=1, name="S2_conv1")
    x = conv_block(x, 64, strides=2, name="S2_conv2")
    x = conv_block(x, 96, strides=1, name="S3_conv1")
    x = conv_block(x, 96, strides=2, name="S3_conv2")
    x = conv_block(x, 128, strides=1, name="S4_conv1")
    x = conv_block(x, 128, strides=2, name="S4_conv2")
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs, name="Final_Verified_TCN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

# ==========================================
# 3. 학습 실행
# ==========================================
model = build_8layer_tcn()
# 새로운 LPN 스케일에 맞춘 모델임을 구분하기 위해 파일명 변경 제안
h5_path = './AI/models/tcn_8layer_combined_final_movenet.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(h5_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("\n🚀 파이프라인 정정 완료! 통합 모델 학습 시작...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=150, 
    batch_size=256, 
    class_weight=class_weights,
    callbacks=callbacks, 
    verbose=1
)

# ==========================================
# 4. 최종 논문 지표 평가 (NTU CV Test)
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"🎉 파이프라인 검증 및 학습 완료!")
print(f"📊 최종 NTU CV Test 정확도: {test_acc*100:.2f}%")
print(f"💾 모델 저장 완료: {h5_path}")
print(f"{'='*60}")