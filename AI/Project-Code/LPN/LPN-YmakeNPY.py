import numpy as np
import os
import glob
import random

def merge_lpn_targets_with_shuffle(npy_dir, save_path, seed=42):
    all_3d_samples = []
    
    # 1. 오직 NTU 스켈레톤 npy 파일만 수집
    npy_files = glob.glob(os.path.join(npy_dir, "*.skeleton.npy"))
    
    # 💡 [안전 장치] OS마다 glob으로 읽어오는 순서가 다를 수 있으므로 무조건 정렬부터 합니다.
    npy_files.sort()
    
    print(f"🔍 발견된 NTU 파일 수: {len(npy_files)}개. 파일 단위 셔플 및 통합을 시작합니다...")

    # 💡 [핵심] 파일 리스트를 무작위로 섞습니다. (60프레임 내부는 그대로 유지됨)
    # 시드(seed)를 42로 고정하여 2D 데이터(X) 병합 시에도 동일한 순서가 되도록 보장합니다.
    random.seed(seed)
    random.shuffle(npy_files)
    print("🔀 파일(비디오) 단위 셔플 완료.")

    for file_path in npy_files:
        try:
            # 2. 파일 로드 (60, 1, 39 형태)
            data = np.load(file_path)
            
            # 데이터 규격 검증
            if data.shape == (60, 1, 39):
                all_3d_samples.append(data)
            else:
                print(f"⚠️ 규격 불일치 건너뜀: {os.path.basename(file_path)} (Shape: {data.shape})")
                
        except Exception as e:
            print(f"❌ 에러 발생 {os.path.basename(file_path)}: {e}")
            continue

    if not all_3d_samples:
        print("❗ 합칠 데이터가 없습니다. 파일 경로와 규격을 확인하세요.")
        return

    # 3. 섞인 비디오 덩어리들을 하나의 거대한 행렬로 합침
    # (N, 60, 1, 39)로 쌓인 뒤, 최종적으로 (N*60, 39) 형태로 펴집니다.
    final_target = np.vstack(all_3d_samples).reshape(-1, 39)
    
    # 저장
    np.save(save_path, final_target)
    print(f"\n✅ LPN용 3D 정답지(Y) 덩어리 셔플 및 통합 완료!")
    print(f"📊 최종 셰이프: {final_target.shape} (총 프레임 수, 39)")
    print(f"💾 저장 위치: {save_path}")

# --- 실행부 ---
npy_folder = './AI/dataset/nturgb+d_skeletons-npy/' 
save_file = './AI/dataset/LPN-label/ntu_lpn_target_3d_shuffled.npy'  # 파일명 변경 추천
merge_lpn_targets_with_shuffle(npy_folder, save_file, seed=42)