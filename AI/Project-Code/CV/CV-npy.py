import numpy as np
import os
import glob

def make_cross_view_dataset_sliding(npy_dir, save_dir, window_size=60, stride=2):
    test_camera = "C001"
    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.skeleton.npy")))
    
    if not npy_files:
        print(f"❌ '{npy_dir}' 폴더에 npy 파일이 없습니다.")
        return

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print(f"📦 Step 2: 슬라이딩 윈도우 생성 및 윈도우별 정규화 시작...")

    for i, file_path in enumerate(npy_files):
        filename = os.path.basename(file_path)
        label = 1 if "A043" in filename else 0
        is_test = test_camera in filename
        
        try:
            # 1단계 결과물 로드: (N, 1, 39)
            raw_data = np.load(file_path) 
            N = raw_data.shape[0]
            if N == 0: continue
            
            # 차원 압축: (N, 1, 39) -> (N, 39)
            raw_data_sq = np.squeeze(raw_data, axis=1)
        except Exception as e:
            print(f"⚠️ {filename} 로드 실패: {e}")
            continue

        # ── 상황 1: 짧은 영상 (Zero-Padding) ──
        if N < window_size:
            window = np.zeros((window_size, 39), dtype=np.float32)
            window[:N] = raw_data_sq
            
            # 윈도우 단위 정규화
            max_dist = np.max(np.abs(window)) + 1e-6
            window_norm = window / max_dist # (60, 39)
            
            if is_test:
                X_test_list.append(window_norm); y_test_list.append(label)
            else:
                X_train_list.append(window_norm); y_train_list.append(label)
        
        # ── 상황 2: 긴 영상 (Sliding Window) ──
        else:
            for start in range(0, N - window_size + 1, stride):
                window = raw_data_sq[start : start + window_size] # (60, 39)
                
                # 윈도우 단위 정규화
                max_dist = np.max(np.abs(window)) + 1e-6
                window_norm = window / max_dist # (60, 39)
                
                if is_test:
                    X_test_list.append(window_norm); y_test_list.append(label)
                else:
                    X_train_list.append(window_norm); y_train_list.append(label)

        if (i + 1) % 5000 == 0:
            print(f"  📂 처리 중... {i+1}/{len(npy_files)}")

    print(f"💾 배열 변환 및 저장 중...")

    # 모든 요소가 (60, 39)이므로 이제 에러 없이 합쳐짐
    X_train = np.array(X_train_list, dtype=np.float32) # (N, 60, 39)
    X_test = np.array(X_test_list, dtype=np.float32)   # (N, 60, 39)
    y_train = np.array(y_train_list, dtype=np.int64).reshape(-1, 1)
    y_test = np.array(y_test_list, dtype=np.int64).reshape(-1, 1)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_cv_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_cv_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_cv_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_cv_test.npy"), y_test)

    print(f"\n{'='*60}")
    print(f"✅ Step 2 완료!")
    print(f"📊 Train 샘플: {X_train.shape}, Test 샘플: {X_test.shape}")
    print(f"{'='*60}")

if __name__ == "__main__":
    make_cross_view_dataset_sliding(
        npy_dir="./AI/dataset/nturgb+d_skeletons-npy/",
        save_dir="./AI/dataset/CV_Split/",
        window_size=60,
        stride=2
    )