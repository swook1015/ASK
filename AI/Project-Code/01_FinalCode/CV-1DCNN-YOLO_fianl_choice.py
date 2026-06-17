import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 💡 1. 환경 설정 (코랄 호환 및 GPU 최적화)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==========================================
# 2. 데이터 로드 (NTU 전용 - 추가 정규화 없음)
# ==========================================
def load_ntu_data_only():
    print("📦 NTU 전용 파이프라인 가동 (Consistency 100%)")
    NTU_DIR = './AI/dataset/CV_Split'
    
    # Step 2에서 이미 Centering + Max-Dist 정규화 완료됨 (-1 ~ 1 범위)
    X_train = np.load(os.path.join(NTU_DIR, 'X_cv_train.npy')).astype(np.float32)
    y_train = np.load(os.path.join(NTU_DIR, 'y_cv_train.npy')).astype(np.float32).reshape(-1, 1)
    
    # Validation은 격리된 Test 카메라(C001) 데이터 사용
    X_val = np.load(os.path.join(NTU_DIR, 'X_cv_test.npy')).astype(np.float32)
    y_val = np.load(os.path.join(NTU_DIR, 'y_cv_test.npy')).astype(np.float32).reshape(-1, 1)

    print(f"📊 스케일 최종 확인: {X_train.min():.2f} ~ {X_train.max():.2f}")
    return X_train, y_train, X_val, y_val

# ==========================================
# 3. 제너레이터 데이터셋 (OOM 방지)
# ==========================================
def create_tf_dataset(X, Y, batch_size=128, is_train=True):
    def generator():
        for i in range(len(X)):
            yield X[i], Y[i]
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(60, 39), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )
    if is_train:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 4. 모델 구성 (8층 TCN - Edge TPU 최적화)
# ==========================================
def conv_block(x, filters, strides=1, name="conv"):
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(filters, 3, strides=strides, padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x) # ReLU6 규격 준수
    return x

def build_8layer_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape, name="input_3d")
    x = conv_block(inputs, 48, strides=1)
    x = conv_block(x, 48, strides=2)
    x = conv_block(x, 64, strides=1)
    x = conv_block(x, 64, strides=2)
    x = conv_block(x, 96, strides=1)
    x = conv_block(x, 96, strides=2)
    x = conv_block(x, 128, strides=1)
    x = conv_block(x, 128, strides=2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)
    
    model = models.Model(inputs, outputs, name="TCN_Pure_NTU_Final")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # 1. 데이터 로드
    X_train, y_train, X_val, y_val = load_ntu_data_only()
    
    # 2. 클래스 가중치 (낙상 데이터 비율 보정)
    neg, pos = np.bincount(y_train.flatten().astype(int))
    class_weights = {0: 1.0, 1: neg / pos}
    print(f"⚖️ 클래스 가중치 적용: 낙상(1)에 {class_weights[1]:.2f}배 가중치 부여")

    # 3. 데이터셋 생성
    train_ds = create_tf_dataset(X_train, y_train, batch_size=128)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=128, is_train=False)

    # 4. 학습 시작
    model = build_8layer_tcn()
    h5_path = './AI/models/tcn_8layer_pure_ntu.h5'

    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(h5_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("\n🚀 NTU 전용 파이프라인 기반 TCN 학습 시작...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1
    )

    print(f"🎉 학습 완료! 모델 저장됨: {h5_path}")