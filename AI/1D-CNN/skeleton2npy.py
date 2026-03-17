#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os
import sys
##  해당 코드는 NTU 공식에 있던 코드 가져옴. 스켈레톤TXT를 NPY로 변환하는 공식코드
##  또한 현재 구조에 맞게 커스터마이징함.
# ==========================================
# ⚙️ 1. 경로 및 설정
# ==========================================
save_npy_path = './AI/dataset/nturgb+d_skeletons-npy/'
load_txt_path = './AI/dataset/nturgb+d_skeletons/'
missing_file_path = './AI/dataset/ntu_rgb120_missings.txt'

# 고정할 프레임 수 (1D-TCN 입력 규격)
TARGET_FRAMES = 60
# MoveNet 매칭 13개 관절 (NTU 0-base 인덱스)
TARGET_INDICES = [3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

# ==========================================
# 🛠️ 2. 유틸리티 함수
# ==========================================
def _load_missing_file(path):
    missing_files = dict()
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                missing_files[line.strip()] = True
    return missing_files

def _print_toolbar(rate, annotation=''):
    toolbar_width = 50
    done = int(toolbar_width * rate)
    sys.stdout.write(f"{annotation} [{'-'*done}{' '*(toolbar_width-done)}] \r")
    sys.stdout.flush()

# ==========================================
# 🧠 3. 핵심 변환 로직 (60프레임 보간 적용)
# ==========================================
def _read_skeleton(file_path):
    with open(file_path, 'r') as f:
        datas = f.readlines()
    
    if not datas: return None
    
    nframe_orig = int(datas[0].strip())
    if nframe_orig == 0: return None
    
    # [원본 프레임 수, 13, 3] 임시 그릇
    temp_skel = np.zeros(shape=(nframe_orig, 13, 3), dtype=np.float32)
    
    cursor = 0
    for frame in range(nframe_orig):
        cursor += 1
        bodycount = int(datas[cursor].strip())
        if bodycount == 0: continue
        
        for body in range(bodycount):
            cursor += 1 # body info
            cursor += 1 # njoints (25)
            for joint in range(25):
                cursor += 1
                if body == 0 and joint in TARGET_INDICES:
                    jointinfo = datas[cursor].strip().split(' ')
                    xyz = np.array(list(map(float, jointinfo[:3])))
                    new_idx = TARGET_INDICES.index(joint)
                    temp_skel[frame, new_idx] = xyz

    # --- [수정 부분 1: 60프레임 선형 보간(Linear Interpolation)] ---
    # 원본 프레임이 몇 개든 상관없이 정확히 60개의 타임스탬프를 생성합니다.
    orig_indices = np.arange(nframe_orig)
    new_indices = np.linspace(0, nframe_orig - 1, TARGET_FRAMES)
    
    skel_60 = np.zeros((TARGET_FRAMES, 13, 3), dtype=np.float32)
    for j in range(13):
        for c in range(3):
            # 시간 축(frame)에 대해 누락되거나 남는 데이터를 부드럽게 메꿈
            skel_60[:, j, c] = np.interp(new_indices, orig_indices, temp_skel[:, j, c])

    # --- [수정 부분 2: 양자화 및 LPN 최적화 후처리] ---
    # 1. 중심점 이동 (Root-Relative): 골반 중심
    root = (skel_60[:, 7:8, :] + skel_60[:, 8:9, :]) / 2
    skel_relative = skel_60 - root

    # 2. 스케일 정규화: Z축 비율 보존하며 -1 ~ 1 사이로 압축
    max_dist = np.max(np.abs(skel_relative)) + 1e-6
    skel_norm = skel_relative / max_dist

    # 3. 저장 형태: [60, 39] (13관절 * 3좌표)
    return skel_norm.reshape(TARGET_FRAMES, 39).astype(np.float32)

# ==========================================
# 🚀 4. 메인 실행부
# ==========================================
if __name__ == '__main__':
    if not os.path.exists(save_npy_path):
        os.makedirs(save_npy_path)
        
    missing_files = _load_missing_file(missing_file_path)
    datalist = [f for f in os.listdir(load_txt_path) if f.endswith('.skeleton')]
    
    print(f"📦 Total files to process: {len(datalist)}")
    
    for ind, each in enumerate(datalist):
        _print_toolbar((ind + 1) / len(datalist), f"({ind+1:5}/{len(datalist):5})")
        
        if each[:20] in missing_files: continue
            
        loadname = os.path.join(load_txt_path, each)
        mat = _read_skeleton(loadname)
        
        if mat is not None:
            save_name = os.path.join(save_npy_path, f"{each}.npy")
            np.save(save_name, mat)

    print("\n✅ 60프레임 고정 및 LPN/1D-TCN용 npy 변환 완료!")