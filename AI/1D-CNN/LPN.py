import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. 데이터 로드 (최적화됨: 이미 합친 npy를 바로 불러옴)
# ==========================================
def load_lpn_dataset_fast(merged_npy_path):
    print(f"📦 통합 데이터셋 로드 중: {merged_npy_path}")
    final_data = np.load(merged_npy_path) # (3412800, 39)
    
    # [Y 데이터] 3D 정답지
    y_train = final_data.astype('float32')
    
    # [X 데이터] 2D 입력값 (Z축만 제거하여 26차원 만들기)
    x_indices = [i for i in range(39) if i % 3 != 2]
    x_train = final_data[:, x_indices].astype('float32')
    
    return x_train, y_train

# ==========================================
# 2. LPN 모델 설계 (논문 참고)
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
# 3. 실행부 (수정 및 최적화됨)
# ==========================================
if __name__ == "__main__":
    # 💡 아까 만들어둔 단일 거대 npy 파일 경로를 사용합니다.
    npy_file_path = './AI/dataset/LPN-label/ntu_lpn_target_3d.npy'
    
    # 1. 데이터 로드
    X, Y = load_lpn_dataset_fast(npy_file_path)
    print(f"✅ 학습 데이터 준비 완료! X: {X.shape}, Y: {Y.shape}")

    # 2. 모델 생성 (오타 수정: build_lpn_model -> build_paper_lpn)
    lpn_model = build_paper_lpn()
    lpn_model.summary()

    # 조기 종료(Early Stopping) 추가: 더 이상 성능 개선이 없으면 100번 다 안 채우고 멈춤
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 3. 학습 시작
    print("🚀 논문 구조 기반 LPN 학습을 시작합니다...")
    history = lpn_model.fit(
        X, Y, 
        epochs=100, 
        batch_size=1024,  # 💡 340만 개 데이터이므로 128은 너무 느립니다. 1024로 키웠습니다.
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # 4. 모델 저장
    os.makedirs('./AI/models', exist_ok=True) # 폴더가 없으면 에러가 나므로 생성 코드 추가
    lpn_model.save('./AI/models/lpn_paper_3d_lifter.h5')
    print("🎉 LPN 모델 저장이 완료되었습니다!")