import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ 1. 경로 및 설정
# ==========================================
DATA_PATH = './skelton2npy/'
BATCH_SIZE = 64
EPOCHS = 30
TARGET_FRAMES = 20

# 13개 관절 인덱스 (코, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목)
# MoveNet에 맞춰서 넣었고, 매칭되지 않은 부분은 전부 삭제함.
CORRECT_JOINT_INDICES = [3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

# ==========================================
# 🧹 2. 전처리 함수 및 제너레이터 
# ==========================================
def preprocess_ntu_data(npy_path):
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        skeleton = data['skel_body0']
        
        # 13개 관절 XY 추출
        skeleton = skeleton[:, CORRECT_JOINT_INDICES, :2]
        
        # 20프레임 고정
        n_frames = skeleton.shape[0]
        if n_frames >= TARGET_FRAMES:
            idx = np.linspace(0, n_frames - 1, TARGET_FRAMES).astype(int)
            skeleton = skeleton[idx]
        else:
            pad_size = TARGET_FRAMES - n_frames
            skeleton = np.concatenate([skeleton, np.tile(skeleton[-1:], (pad_size, 1, 1))], axis=0)
            
        # (20, 26) 평탄화 및 정규화
        skeleton = skeleton.reshape(TARGET_FRAMES, -1)
        denom = np.max(skeleton) - np.min(skeleton) + 1e-6
        return (skeleton - np.min(skeleton)) / denom
    except:
        return None

class NTUGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size=32):
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = [], []
        for f in batch_files:
            feat = preprocess_ntu_data(os.path.join(DATA_PATH, f))
            if feat is not None:
                X.append(feat)
                y.append(1 if 'A043' in f else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Edge TPU 최적화용 4D 변환 (Batch, 20, 26, 1)
        if len(X) > 0:
            X = np.expand_dims(X, axis=-1)
            
        return X, y

# 파일 스캔 및 분할
print("📁 파일 스캔 중...")
all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_gen = NTUGenerator(train_files, batch_size=BATCH_SIZE)
val_gen = NTUGenerator(val_files, batch_size=BATCH_SIZE)

# ==========================================
# 🧠 3. 전이학습을 고려한 Edge TPU 모델 정의
# ==========================================

def create_pretrained_model(batch_size=None):
    inputs = layers.Input(batch_shape=(batch_size, TARGET_FRAMES, 26, 1), name='input')
    
    # [특징 추출기 (Feature Extractor)] - 나중에 URFD 학습 시 Freeze 할 부분
    x = layers.Conv2D(32, kernel_size=(3, 26), activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 1), name='pool1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 1), activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name='pool2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    
    # [분류기 (Classifier)] - 나중에 URFD 학습 시 집중적으로 학습시킬 부분
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(32, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

# ==========================================
# 🥇 4. 기초 학습 (Pre-training) 실행
# ==========================================
print("\n🚀 NTU 데이터셋 기초 학습(Pre-training) 시작...")
base_model = create_pretrained_model()
base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 콜백 설정 (가장 좋은 가중치 저장 및 조기 종료)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('ntu_pretrained_base.h5', monitor='val_loss', save_best_only=True)

# NTU의 극심한 데이터 불균형 해결 (A043 낙상에 가중치 50배)
class_weight_dict = {0: 1.0, 1: 50.0}

base_model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=EPOCHS, 
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight_dict
)

print("\n✅ 기초 학습 완료! 모델이 'ntu_pretrained_base.h5'로 안전하게 저장되었습니다.")

# ⚠️ 주의: 전이학습을 할 거라면, 지금 TFLite 양자화를 할 필요가 없습니다!
# 나중에 URFD 데이터로 최종 파인튜닝(Fine-tuning)이 끝난 뒤에 TFLite로 변환해야 합니다.