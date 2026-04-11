import os
# 💡 핵심: 텐서플로우를 불러오기 전에 반드시 선언해야 합니다.
# 이 설정이 있어야 모델이 Keras 2 Legacy 규격으로 저장됩니다.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
# Legacy 환경에서는 tf.keras가 아닌 별도의 keras 패키지가 권장되기도 하지만,
# 환경 변수 설정만으로도 tf.keras가 Legacy 모드로 동작합니다.
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. 모델 설계 (TimeDistributed + Residual Block)
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
    
    model = models.Model(inputs, outputs, name="LPN_Sequence_Lifter_Legacy")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==========================================
# 2. 데이터 로드 및 시계열 변환
# ==========================================
def load_and_reshape_data(x_path, y_path, seq_len=60):
    print("📦 데이터 로드 중...")
    X_raw = np.load(x_path).astype('float32')
    Y_raw = np.load(y_path).astype('float32')
    
    num_samples = len(X_raw) // seq_len
    X = X_raw[:num_samples * seq_len].reshape(-1, seq_len, 26)
    Y = Y_raw[:num_samples * seq_len].reshape(-1, seq_len, 39)
    
    return X, Y

# ==========================================
# 3. 메인 실행부
# ==========================================
if __name__ == "__main__":
    x_file = './AI/dataset/LPN-train/ntu_lpn_input_2d_final.npy'
    y_file = './AI/dataset/LPN-label/ntu_lpn_target_3d_shuffled.npy'
    
    X, Y = load_and_reshape_data(x_file, y_file, seq_len=60)
    print(f"✅ 데이터 변환 완료! X: {X.shape}, Y: {Y.shape}")

    model = build_lpn_sequence_model(seq_len=60)
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    print("🚀 Legacy 모드로 LPN 학습 시작...")
    model.fit(
        X, Y,
        epochs=100,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop, lr_reducer],
        verbose=1
    )

    os.makedirs('./AI/models', exist_ok=True)
    # 💡 Keras 2 규격의 .h5 파일로 저장
    model.save('./AI/models/lpn_remaster_60_legacy.h5')
    print("🎉 Keras 2 Legacy 모델 저장 완료!")