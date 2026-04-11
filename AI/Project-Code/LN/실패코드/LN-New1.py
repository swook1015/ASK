import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 💡 1. 환경 설정 (코랄 호환 및 GPU 최적화)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# GPU 메모리 점진적 할당 설정 (4GB VRAM 효율적 사용)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 메모리 점진적 할당 설정 완료")
    except RuntimeError as e:
        print(f"❌ GPU 설정 오류: {e}")

# ==========================================
# 2. 모델 설계 (LPN Sequence Lifter)
# ==========================================
def residual_block_td(x, nodes, dropout_rate=0.5):
    shortcut = x
    x = layers.TimeDistributed(layers.Dense(nodes))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    return layers.Add()([shortcut, x])

def build_lpn_model(seq_len=60, input_dim=26, output_dim=39):
    inputs = layers.Input(shape=(seq_len, input_dim), name="input_2d")
    
    # Initial Dense Projection
    x = layers.TimeDistributed(layers.Dense(1024))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    
    # Residual Blocks
    x = residual_block_td(x, 1024)
    x = residual_block_td(x, 1024)
    
    # 3D Coordinates Output
    outputs = layers.TimeDistributed(layers.Dense(output_dim), name="output_3d")(x)
    
    model = models.Model(inputs, outputs, name="LPN_HipRoot_Final")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# ==========================================
# 3. 데이터 로드 및 파이프라인 구축 (OOM 방지)
# ==========================================
def load_and_preprocess(npy_path):
    print(f"📦 데이터 로딩 중: {os.path.basename(npy_path)}")
    raw_data = np.load(npy_path).astype(np.float32)
    N, T, _ = raw_data.shape
    
    # [Y] 정답지: 3D 전체 (x, y, z)
    Y = raw_data
    
    # [X] 입력값: 2D만 추출 (x, y)
    data_reshaped = raw_data.reshape(N, T, 13, 3)
    X = data_reshaped[:, :, :, 0:2].reshape(N, T, 26)
    
    print(f"🔍 [스케일 체크] X: {X.min():.2f}~{X.max():.2f} / Y: {Y.min():.2f}~{Y.max():.2f}")
    return X, Y

def create_tf_dataset(X, Y, batch_size=128, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if is_train:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ==========================================
# 4. 학습 실행
# ==========================================
if __name__ == "__main__":
    # 경로 설정
    TRAIN_PATH = './AI/dataset/CV_Split/X_cv_train.npy'
    TEST_PATH = './AI/dataset/CV_Split/X_cv_test.npy'

    # 데이터 로드 (System RAM에 로드)
    X_train, Y_train = load_and_preprocess(TRAIN_PATH)
    X_val, Y_val = load_and_preprocess(TEST_PATH)

    # Dataset 생성 (GPU로 조금씩 쏴주기 위해)
    BATCH_SIZE = 128  # RTX 2060 4GB에 안전한 사이즈
    train_ds = create_tf_dataset(X_train, Y_train, batch_size=BATCH_SIZE)
    val_ds = create_tf_dataset(X_val, Y_val, batch_size=BATCH_SIZE, is_train=False)

    model = build_lpn_model()
    
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
        callbacks.ModelCheckpoint('./AI/models/lpn_hip_root_best.h5', monitor='val_loss', save_best_only=True)
    ]

    print("\n🚀 GPU 가속 LPN 학습 시작...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        callbacks=cbs,
        verbose=1
    )

    # 최종 모델 저장
    os.makedirs('./AI/models', exist_ok=True)
    model.save('./AI/models/lpn_hip_root_final.h5')
    print("🎉 LPN 학습 및 저장 완료!")