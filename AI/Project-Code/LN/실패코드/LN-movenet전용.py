import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# (1. 모델 설계 부분은 기존과 동일하므로 생략)
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
# 2. 데이터 로드 및 최적화된 정규화 (Centering & Scaling)
# ==========================================
def load_and_normalize_lpn_data(raw_3d_path):
    print(f"📦 원본 CV 3D 데이터 로드 중... ({os.path.basename(raw_3d_path)})")
    raw_3d = np.load(raw_3d_path).astype(np.float32)
    
    N, frames, _ = raw_3d.shape
    raw_3d_reshaped = raw_3d.reshape(N, frames, 13, 3)
    
    # 🎯 1단계: 영점 조절 (골반 기준)
    root_xyz = raw_3d_reshaped[:, :, 0:1, :] 
    centered_3d = raw_3d_reshaped - root_xyz 
    
    # 🔥 2단계: 스케일 정규화 (치명적 오류 방지)
    # 프레임별로 인물 크기가 다를 수 있으므로, 골반에서 가장 먼 관절 거리를 구해 전체 좌표를 나눔
    # (이렇게 하면 모든 값이 대략 -1.0 ~ 1.0 사이로 맞춰짐)
    max_dist = np.max(np.linalg.norm(centered_3d, axis=-1), axis=-1, keepdims=True)
    max_dist = np.expand_dims(max_dist, axis=-1)
    # 0으로 나누는 것 방지
    max_dist[max_dist == 0] = 1e-6 
    
    normalized_3d = centered_3d / max_dist

    # 정답지 Y 생성 (39 묶음)
    Y = normalized_3d.reshape(N, frames, 39)
    
    # 입력 X 생성 (Z축 제거)
    normalized_2d = normalized_3d[:, :, :, 0:2] 
    X = normalized_2d.reshape(N, frames, 26) 
    
    return X, Y

# (3. 메인 실행부는 기존과 완벽히 동일하므로 그대로 사용하시면 됩니다.)
if __name__ == "__main__":
    # 💡 아까 1D-TCN에서 썼던 CV 분할 데이터를 그대로 가져옵니다!
    train_path = './AI/dataset/CV_Split/X_cv_train.npy'
    val_path = './AI/dataset/CV_Split/X_cv_test.npy'
    
    # Train, Val 각각 정규화를 거쳐서 X(2D), Y(3D) 생성
    X_train, Y_train = load_and_normalize_lpn_data(train_path)
    X_val, Y_val = load_and_normalize_lpn_data(val_path)
    
    print(f"\n✅ 데이터 변환 및 영점 조절 완료!")
    print(f"📊 Train -> X(2D): {X_train.shape}, Y(3D): {Y_train.shape}")
    print(f"📊 Val   -> X(2D): {X_val.shape}, Y(3D): {Y_val.shape}")

    model = build_lpn_sequence_model(seq_len=60, input_dim=26, output_dim=39)
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    print("\n🚀 논문 프로토콜(CV) 기반 LN 학습 시작...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val), # 🔥 섞지 않고 완벽히 격리된 Test 카메라(-45도) 데이터 사용
        epochs=150,
        batch_size=128,
        callbacks=[early_stop, lr_reducer],
        verbose=1,
        shuffle=True # 학습용 Train 데이터 내부에서만 섞음
    )

    os.makedirs('./AI/models', exist_ok=True)
    # 💡 Keras 2 규격의 .h5 파일로 저장 (파일명에 cv 명시)
    save_path = './AI/models/lpn_cv_60_legacy_movenet.h5'
    model.save(save_path)
    print(f"🎉 Keras 2 Legacy 모델 저장 완료: {save_path}")