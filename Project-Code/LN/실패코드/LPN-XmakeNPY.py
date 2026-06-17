import numpy as np
import os

def prepare_lpn_data_from_ntu(merged_3d_path, save_dir):
    print(f"📦 NTU 통합 3D 데이터 로드 중: {merged_3d_path}")
    
    # 1. 원본 3D 데이터 로드 (Shape: Total_Frames, 39)
    # 이미 13개 관절로 가공된 성욱님의 데이터를 가져옵니다.
    y_train = np.load(merged_3d_path).astype(np.float32)
    
    # 2. 2D 입력 데이터(X) 생성: Z축만 제거
    # [x1, y1, z1, x2, y2, z2 ...] 구조에서 인덱스 2, 5, 8... (3의 배수-1)을 제외합니다.
    xy_indices = [i for i in range(39) if i % 3 != 2]
    x_train = y_train[:, xy_indices] # Shape: (Total_Frames, 26)
    
    # 3. 데이터 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    x_save_path = os.path.join(save_dir, 'ntu_lpn_input_2d_final.npy')
    y_save_path = os.path.join(save_dir, 'ntu_lpn_target_3d_final.npy') # 관리용 복사본
    
    np.save(x_save_path, x_train)
    np.save(y_save_path, y_train)
    
    # 4. 양자화용 대표 데이터셋 (앞부분 500개 샘플)
    rep_path = x_save_path.replace('.npy', '_rep.npy')
    np.save(rep_path, x_train[:500])
    
    print(f"\n✅ LPN 학습 세트 준비 완료!")
    print(f"📊 입력(X) 셰이프: {x_train.shape} (2D 좌표 26개)")
    print(f"📊 정답(Y) 셰이프: {y_train.shape} (3D 좌표 39개)")
    print(f"💾 저장 위치: {save_dir}")

# 실행 (경로는 성욱님의 환경에 맞게 수정하세요)
target_path = './AI/dataset/LPN-label/ntu_lpn_target_3d_shuffled.npy'
save_folder = './AI/dataset/LPN-train/'
prepare_lpn_data_from_ntu(target_path, save_folder)