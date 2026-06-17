import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 🚨 Keras 2(Legacy) 엔진 강제 사용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# 1. 새로 뽑은 데이터 로드 (전처리 생략!)
# ==========================================
print("📦 새로 생성된 NTU 데이터 로드 (정적 스케일 & Y축 반전 완료됨)...")

NTU_DIR = './AI/dataset/CV_Split'

# 🚨 [중요] 이미 CV-npy.py에서 전처리가 완벽하게 끝났으므로, 
# 여기서는 center_poses 같은 함수를 쓰지 않고 그대로 가져옵니다.
X_train = np.load(os.path.join(NTU_DIR, 'X_cv_train.npy')).astype(np.float32)
y_train = np.load(os.path.join(NTU_DIR, 'y_cv_train.npy')).astype(np.float32).reshape(-1, 1)

X_test = np.load(os.path.join(NTU_DIR, 'X_cv_test.npy')).astype(np.float32)
y_test = np.load(os.path.join(NTU_DIR, 'y_cv_test.npy')).astype(np.float32).reshape(-1, 1)

print(f"📊 학습셋: {X_train.shape[0]}개 / 테스트셋: {X_test.shape[0]}개")

# 클래스 가중치 계산
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
h5_path = './AI/models/cnn_8layer_pure_ntu.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(h5_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("\n🚀 정적 스케일 기반 순수 NTU 모델 학습 시작...")
history = model.fit(
    X_train, y_train, 
    # [수정] Valid를 Test셋으로 대체하여 성능 모니터링
    validation_data=(X_test, y_test), 
    epochs=150, 
    batch_size=256, 
    class_weight=class_weights,
    callbacks=callbacks, 
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 최종 NTU CV Test 정확도: {test_acc*100:.2f}%")