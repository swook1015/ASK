import numpy as np
import os
import glob

def merge_lpn_targets(npy_dir, save_path):
    all_3d_samples = []
    
    # 1. 오직 NTU 스켈레톤 npy 파일만 수집 (coco 파일 등 제외)
    npy_files = glob.glob(os.path.join(npy_dir, "*.skeleton.npy"))
    print(f"🔍 발견된 NTU 파일 수: {len(npy_files)}개. 통합을 시작합니다...")

    for file_path in npy_files:
        try:
            # 2. 파일 로드 (이미 60, 39 형태임)
            data = np.load(file_path)
            
            # 데이터 규격 검증 (안전장치)
            if data.shape == (60, 39):
                all_3d_samples.append(data)
            else:
                print(f"⚠️ 규격 불일치 건너뜀: {os.path.basename(file_path)} (Shape: {data.shape})")
                
        except Exception as e:
            print(f"❌ 에러 발생 {os.path.basename(file_path)}: {e}")
            continue

    if not all_3d_samples:
        print("❗ 합칠 데이터가 없습니다. 파일 경로와 규격을 확인하세요.")
        return

    # 3. 모든 프레임을 하나의 거대한 행렬로 합침
    final_target = np.vstack(all_3d_samples)
    
    # 저장
    np.save(save_path, final_target)
    print(f"\n✅ LPN용 3D 정답지(Y) 통합 완료!")
    print(f"📊 최종 셰이프: {final_target.shape} (총 프레임 수, 39)")
    print(f"💾 저장 위치: {save_path}")

# --- 실행부 ---
npy_folder = './AI/dataset/nturgb+d_skeletons-npy/' # 개별 npy들이 저장된 폴더
save_file = './AI/dataset/LPN-label/ntu_lpn_target_3d.npy'   # 최종 합본 파일명
merge_lpn_targets(npy_folder, save_file)