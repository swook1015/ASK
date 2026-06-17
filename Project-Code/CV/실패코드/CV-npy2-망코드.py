import numpy as np
import os
import glob

def make_cross_view_dataset_sliding(npy_dir, save_dir, window_size=60, stride=2):
    test_camera = "C001"
    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.skeleton.npy")))
    
    # 🚨 [핵심 1] 모든 데이터를 동일하게 나누는 정적 스케일링 상수
    STATIC_SCALE = 480.0 

    if not npy_files:
        print(f"❌ '{npy_dir}' 폴더에 npy 파일이 없습니다.")
        return

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print(f"📦 Step 2: Y축 반전 및 정적 스케일링 슬라이딩 윈도우 생성...")

    for i, file_path in enumerate(npy_files):
        filename = os.path.basename(file_path)
        label = 1 if "A043" in filename else 0
        is_test = test_camera in filename
        
        try:
            raw_data = np.load(file_path) 
            N = raw_data.shape[0]
            if N == 0: continue
            
            raw_data_sq = np.squeeze(raw_data, axis=1) # (N, 39)
            
            # 🚨 [핵심 2] Y축 반전 (이미지 좌표계 -> 수학 좌표계)
            # 인덱스 1, 4, 7... 에 해당하는 Y좌표만 부호 반전
            raw_data_sq[:, 1::3] = -raw_data_sq[:, 1::3]

            if N < window_size: continue

            for start in range(0, N - window_size + 1, stride):
                window = raw_data_sq[start : start + window_size]
                
                # 🚨 [핵심 3] 동적 max_dist 완전 제거 및 정적 스케일링 적용
                window_norm = window / STATIC_SCALE 
                
                if is_test:
                    X_test_list.append(window_norm); y_test_list.append(label)
                else:
                    X_train_list.append(window_norm); y_train_list.append(label)

        except Exception as e:
            pass

        if (i + 1) % 5000 == 0:
            print(f"  📂 처리 중... {i+1}/{len(npy_files)}")

    print(f"💾 배열 변환 및 저장 중...")
    X_train = np.array(X_train_list, dtype=np.float32) 
    X_test = np.array(X_test_list, dtype=np.float32)   
    y_train = np.array(y_train_list, dtype=np.int64).reshape(-1, 1)
    y_test = np.array(y_test_list, dtype=np.int64).reshape(-1, 1)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_cv_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_cv_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_cv_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_cv_test.npy"), y_test)
    print("✅ 데이터 정렬 완료. 이제 모델 재학습을 진행하세요.")

if __name__ == '__main__':
    npy_dir = './AI/dataset/nturgb+d_skeletons-npy/'
    save_dir = './AI/dataset/CV_Split'
    make_cross_view_dataset_sliding(npy_dir, save_dir)