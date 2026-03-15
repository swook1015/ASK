import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
print("📦 데이터 불러오는 중...")
X = np.load('ntu_X_train.npy')
y = np.load('ntu_y_train.npy')

# 2. 학습용(80%) / 테스트용(20%) 분리 (낙상 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ 학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# 3. 데이터 불균형 해결 (Class Weight)
neg = len(y_train[y_train == 0])
pos = len(y_train[y_train == 1])
total = neg + pos

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"⚖️ 클래스 가중치 적용 - 정상: {weight_for_0:.2f}, 낙상: {weight_for_1:.2f}")

# ---------------------------------------------------------
# 🧠 4. 논문 구조를 100% 반영한 TCN 모델 설계
# ---------------------------------------------------------
def build_paper_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape)
    
    # [STAGE 2: Fall Detection Network (TCN)]
    # TCN Block 1: Dilation 1
    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x) # 학습 안정성을 위해 추가
    
    # TCN Block 2: Dilation 2
    x = layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # TCN Block 3: Dilation 4
    x = layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # [STAGE 3: Classifier (최종 판별)]
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x) # 60배 많은 정상 데이터에 과적합되지 않도록 방지
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Final Output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

model = build_paper_tcn()
model.summary()

# 5. 콜백 설정 (auc가 15번 안 오르면 중단)
callbacks = [
    EarlyStopping(monitor='val_auc', patience=15, mode='max', verbose=1, restore_best_weights=True),
    ModelCheckpoint('tcn_ntu_paper_model.h5', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# 6. 학습 시작!
print("🔥 논문 구조 기반 1D-TCN 기초학습 시작!")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    class_weight=class_weight,
    callbacks=callbacks
)

print("🎉 기초학습 완료! 'tcn_ntu_paper_model.h5'가 저장되었습니다.")