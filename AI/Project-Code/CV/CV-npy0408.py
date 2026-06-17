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

    print(f"📦 Step 2: 슬라이딩 윈도우 생성 (추가 정규화 없음)...")

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

            # 슬라이딩 윈도우 (60프레임 단위)
            if N < window_size:
                padded = np.zeros((window_size, 39), dtype=np.float32)
                padded[:N] = raw_data_sq
                
                # 🚨 수정됨: 이중 정규화(window / max_dist) 로직 완전 삭제
                if is_test:
                    X_test_list.append(padded); y_test_list.append(label)
                else:
                    X_train_list.append(padded); y_train_list.append(label)
            else:
                for start in range(0, N - window_size + 1, stride):
                    window = raw_data_sq[start : start + window_size]
                    
                    # 🚨 수정됨: 이중 정규화 로직 완전 삭제
                    if is_test:
                        X_test_list.append(window); y_test_list.append(label)
                    else:
                        X_train_list.append(window); y_train_list.append(label)

        except Exception as e:
            print(f"⚠️ 에러 발생 ({filename}): {e}")

        if (i + 1) % 5000 == 0:
            print(f"  📂 처리 중... {i+1}/{len(npy_files)}")

    print(f"💾 배열 변환 및 저장 중...")

    X_train = np.array(X_train_list, dtype=np.float32) # (N, 60, 39)
    X_test = np.array(X_test_list, dtype=np.float32)   # (N, 60, 39)
    y_train = np.array(y_train_list, dtype=np.int64).reshape(-1, 1)
    y_test = np.array(y_test_list, dtype=np.int64).reshape(-1, 1)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_cv_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_cv_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_cv_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_cv_test.npy'), y_test)

    print(f"✅ 완료! Train: {X_train.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    npy_folder = './AI/dataset/nturgb+d_skeletons-npy/'
    save_folder = './AI/dataset/CV_Split/'
    make_cross_view_dataset_sliding(npy_folder, save_folder)