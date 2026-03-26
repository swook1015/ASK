import numpy as np
import os
import glob
import random

def make_cross_subject_dataset(npy_dir, save_dir):
    # 1. 논문 기준 학습용 Subject ID (파일명의 S001 형태) 
    train_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    train_patterns = [f"S{i:03d}" for i in train_ids]

    npy_files = glob.glob(os.path.join(npy_dir, "*.skeleton.npy"))
    npy_files.sort()

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print(f"🔍 총 {len(npy_files)}개 파일 분석 시작...")

    for file_path in npy_files:
        filename = os.path.basename(file_path)
        
        # 데이터 로드 (60, 1, 39)
        try:
            data = np.load(file_path)
            if data.shape != (60, 1, 39): continue
            
            # 라벨 결정 (A043 이면 낙상: 1, 아니면 0)
            label = 1 if 'A043' in filename else 0
            
            # 2. 논문 기준에 따른 분류 (S 번호 확인) 
            is_train = any(pattern in filename for pattern in train_patterns)
            
            if is_train:
                X_train_list.append(data)
                y_train_list.append(label)
            else:
                X_test_list.append(data)
                y_test_list.append(label)
                
        except: continue

    # 배열 변환 및 저장
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list).reshape(-1, 1)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list).reshape(-1, 1)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_cs_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_cs_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_cs_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_cs_test.npy'), y_test)

    print(f"✅ CS 데이터셋 생성 완료!")
    print(f"📊 Train: {len(X_train)} samples / Test: {len(X_test)} samples")

# 실행
make_cross_subject_dataset('./AI/dataset/nturgb+d_skeletons-npy/', './AI/dataset/CS_Split/')