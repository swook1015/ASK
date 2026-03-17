import numpy as np
import os

def prepare_lpn_data_from_ntu(merged_3d_path, save_dir):
    print(f"📦 NTU 통합 3D 데이터 로드 중: {merged_3d_path}")
    
    # 1. 원본 3D 데이터 로드 및 Reshape (N, 13관절, 3차원)
    raw_data = np.load(merged_3d_path).astype(np.float32)
    N = raw_data.shape[0]
    raw_data = raw_data.reshape(N, 13, 3)
    
    # ==========================================
    # 💡 핵심 추가: Root-Relative 및 정규화
    # ==========================================
    # 1) Root-Relative: 골반(7, 8번 관절)의 중앙점을 (0,0,0)으로!
    root_3d = (raw_data[:, 7, :] + raw_data[:, 8, :]) / 2.0
    root_3d = np.expand_dims(root_3d, axis=1) # (N, 1, 3)
    centered_data = raw_data - root_3d
    
    # 2) Scale Normalization: 가장 멀리 뻗은 관절 기준으로 -1 ~ 1 압축
    max_dist = np.max(np.abs(centered_data), axis=(1, 2), keepdims=True) + 1e-6
    normalized_data = centered_data / max_dist
    # ==========================================
    
    # 2. 2D 입력 데이터(X) 생성: Z축만 제거 (X, Y만 남김)
    x_train_2d = normalized_data[:, :, :2].reshape(N, 26) # (N, 26)
    y_train_3d = normalized_data.reshape(N, 39)           # (N, 39)
    
    # 3. 데이터 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    x_save_path = os.path.join(save_dir, 'ntu_lpn_input_2d.npy')
    y_save_path = os.path.join(save_dir, 'ntu_lpn_target_3d_final.npy')
    
    np.save(x_save_path, x_train_2d)
    np.save(y_save_path, y_train_3d)
    
    # 4. 양자화용 대표 데이터셋 (순서를 섞어서 500개 추출하면 더 좋습니다)
    indices = np.arange(N)
    np.random.shuffle(indices)
    rep_data = x_train_2d[indices[:500]]
    
    rep_path = x_save_path.replace('.npy', '_rep.npy')
    np.save(rep_path, rep_data)
    
    print(f"\n✅ LPN 정규화 학습 세트 준비 완료!")
    print(f"📊 입력(X) 셰이프: {x_train_2d.shape} | 범위: [{np.min(x_train_2d):.2f} ~ {np.max(x_train_2d):.2f}]")
    print(f"💾 저장 위치: {save_dir}")

# 실행
target_path = './AI/dataset/LPN-label/ntu_lpn_target_3d.npy'
save_folder = './AI/dataset/LPN-train/'
prepare_lpn_data_from_ntu(target_path, save_folder)