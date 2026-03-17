import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. 데이터 로드 (수정됨: 이미 만들어진 2D와 3D 파일을 각각 불러옴)
# ==========================================
def load_lpn_dataset_direct(x_path, y_path):
    print(f"📦 2D 입력(X) 데이터 로드 중: {x_path}")
    # 이미 26차원(Z축 제거됨)으로 가공된 문제지 파일 로드
    x_train = np.load(x_path).astype('float32') 
    
    print(f"📦 3D 정답(Y) 데이터 로드 중: {y_path}")
    # 원본 39차원 정답지 파일 로드
    y_train = np.load(y_path).astype('float32') 
    
    return x_train, y_train

# ==========================================
# 2. LPN 모델 설계 (논문 참고 구조 그대로 유지)
# ==========================================
def residual_block(x, nodes, dropout_rate=0.5):
    shortcut = x
    # 첫 번째 Dense + BN + ReLU + Dropout
    x = layers.Dense(nodes)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 두 번째 Dense + BN + ReLU
    x = layers.Dense(nodes)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Skip Connection (더하기)
    return layers.Add()([shortcut, x])

def build_paper_lpn(input_dim=26, output_dim=39):
    inputs = layers.Input(shape=(input_dim,))
    
    # STAGE 1: Linear(26 -> 1024)
    x = layers.Dense(1024)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Residual Block 1 & 2
    x = residual_block(x, 1024)
    x = residual_block(x, 1024)
    
    # Output Layer: Linear(1024 -> 39)
    outputs = layers.Dense(output_dim)(x)
    
    model = models.Model(inputs, outputs, name="LPN_Paper_Structure")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==========================================
# 3. 실행부 
# ==========================================
if __name__ == "__main__":
    # 💡 아까 만들어둔 두 개의 파일 경로를 각각 지정합니다.
    x_file_path = './AI/dataset/LPN-train/ntu_lpn_input_2d.npy'
    y_file_path = './AI/dataset/LPN-train/ntu_lpn_target_3d_final.npy'
    
    # 1. 데이터 로드 (자르는 로직 없이 바로 불러옴!)
    X, Y = load_lpn_dataset_direct(x_file_path, y_file_path)
    print(f"✅ 학습 데이터 준비 완료! X(문제): {X.shape}, Y(정답): {Y.shape}")

    # 2. 모델 생성
    lpn_model = build_paper_lpn()
    lpn_model.summary()

    # 콜백 설정 (조기 종료 + 학습률 자동 조절)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # 3. 학습 시작
    print("🚀 논문 구조 기반 LPN 학습을 시작합니다...")
    history = lpn_model.fit(
        X, Y, 
        epochs=100, 
        batch_size=1024,  # 대용량 데이터이므로 배치 사이즈를 넉넉하게!
        validation_split=0.1,
        callbacks=[early_stop, lr_reducer],
        verbose=1
    )

    # 4. 모델 저장
    os.makedirs('./AI/models', exist_ok=True)
    save_path = './AI/models/lpn_paper_3d_lifter.h5'
    lpn_model.save(save_path)
    print(f"🎉 LPN 모델 저장이 완료되었습니다! (저장 위치: {save_path})")