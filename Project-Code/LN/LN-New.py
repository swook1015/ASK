import os
# 💡 코랄 보드 호환을 위해 텐서플로우 불러오기 전 선언
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# GPU 메모리 점진적 할당 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==========================================
# 1. 모델 설계 (논문 구조 유지)
# ==========================================
def residual_block_td(x, nodes, dropout_rate=0.5):
    shortcut = x
    x = layers.TimeDistributed(layers.Dense(nodes))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    return layers.Add()([shortcut, x])

def build_lpn_sequence_model(seq_len=60, input_dim=26, output_dim=39):
    inputs = layers.Input(shape=(seq_len, input_dim))
    x = layers.TimeDistributed(layers.Dense(1024))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    x = residual_block_td(x, 1024)
    x = residual_block_td(x, 1024)
    outputs = layers.TimeDistributed(layers.Dense(output_dim))(x)
    
    model = models.Model(inputs, outputs, name="LPN_Sequence_Lifter_CV")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==========================================
# 2. 데이터 로드 및 제너레이터 (OOM 해결 핵심)
# ==========================================
def load_lpn_data_only(path):
    print(f"📦 데이터 로드 중: {os.path.basename(path)}")
    data = np.load(path).astype(np.float32)
    # Step 2에서 이미 골반 영점/정규화 완료됨
    N, T, _ = data.shape
    Y = data
    X = data.reshape(N, T, 13, 3)[:, :, :, :2].reshape(N, T, 26)
    return X, Y

def get_dataset(X, Y, batch_size=64, is_train=True):
    # 💡 한꺼번에 GPU에 올리지 않고 한 주먹(Batch)씩만 전달
    def generator():
        for i in range(len(X)):
            yield X[i], Y[i]
            
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(60, 26), dtype=tf.float32),
            tf.TensorSpec(shape=(60, 39), dtype=tf.float32)
        )
    )
    if is_train:
        ds = ds.shuffle(5000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. 메인 실행부
# ==========================================
if __name__ == "__main__":
    train_path = './AI/dataset/CV_Split/X_cv_train.npy'
    val_path = './AI/dataset/CV_Split/X_cv_test.npy'

    # 1. 램(RAM)으로 데이터 로드
    X_train, Y_train = load_lpn_data_only(train_path)
    X_val, Y_val = load_lpn_data_only(val_path)

    # 2. GPU용 징검다리(Dataset) 생성
    BATCH_SIZE = 64 # RTX 2060 안전빵 수치
    train_ds = get_dataset(X_train, Y_train, batch_size=BATCH_SIZE)
    val_ds = get_dataset(X_val, Y_val, batch_size=BATCH_SIZE, is_train=False)

    model = build_lpn_sequence_model()
    
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    print("\n🚀 제너레이터 파이프라인으로 LPN 학습 시작...")
    model.fit(
        train_ds, # 넘파이 직접 안 넣고 데이터셋 객체 넣음
        validation_data=val_ds,
        epochs=150,
        callbacks=cbs,
        verbose=1
    )

    os.makedirs('./AI/models', exist_ok=True)
    save_path = './AI/models/0408new_lpn_cv_60_legacy.h5'
    model.save(save_path)
    print(f"🎉 모델 저장 완료: {save_path}")