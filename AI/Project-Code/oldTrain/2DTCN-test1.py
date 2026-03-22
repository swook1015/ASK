import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_coral_friendly_tcn(input_shape=(60, 39)):
    inputs = layers.Input(shape=input_shape)
    
    # 1. 초기 진입
    x = layers.Conv1D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 2. Stride를 이용한 단계적 압축 (Dilation 대신 사용)
    # 코랄 보드가 가장 잘 처리하는 구조입니다.
    x = layers.Conv1D(64, 3, strides=2, padding='same')(x) # 60 -> 30
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(128, 3, strides=2, padding='same')(x) # 30 -> 15
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(256, 3, strides=2, padding='same')(x) # 15 -> 8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 3. 출력층
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='fall_output')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 학습 로직은 동일하게 진행
model = build_coral_friendly_tcn()
model.summary()