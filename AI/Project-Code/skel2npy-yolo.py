import numpy as np
import os
import sys

# ==========================================
# ⚙️ 1. 경로 및 설정
# ==========================================
save_npy_path = './AI/dataset/nturgb+d_skeletons-npy/'
load_txt_path = './AI/dataset/nturgb+d_skeletons/'
missing_file_path = './AI/dataset/ntu_rgb120_missings.txt'

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
# 🧠 2. 핵심 변환 로직 (골반 영점 & 정규화 제외)
# ==========================================
def _read_skeleton(file_path):
    with open(file_path, 'r') as f:
        datas = f.readlines()
    if not datas: return None
    nframe_orig = int(datas[0].strip())
    if nframe_orig == 0: return None
    
    # YOLO 매핑 (0번은 골반중심으로 따로 넣음)
    ntu_to_yolo_idx = [4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]
    temp_skel = np.zeros(shape=(nframe_orig, 13, 3), dtype=np.float32)
    
    cursor = 0
    for frame in range(nframe_orig):
        cursor += 1
        bodycount = int(datas[cursor].strip())
        if bodycount == 0: continue
        for body in range(bodycount):
            cursor += 2 
            joints_25 = np.zeros((25, 3), dtype=np.float32)
            for joint in range(25):
                cursor += 1
                if body == 0:
                    jointinfo = datas[cursor].strip().split(' ')
                    joints_25[joint] = np.array(list(map(float, jointinfo[:3])))
            
            if body == 0:
                # 🔥 0번 자리에 골반 중심 삽입
                hip_center = (joints_25[12] + joints_25[16]) / 2.0
                temp_skel[frame, 0] = hip_center
                for i, n_idx in enumerate(ntu_to_yolo_idx):
                    temp_skel[frame, i+1] = joints_25[n_idx]

    # 💡 [핵심] 정규화 없이 센터링(Relative)만 수행
    root = temp_skel[:, 0:1, :] 
    skel_relative = temp_skel - root

    # 저장 형태: (프레임수, 1, 39)
    return skel_relative.reshape(nframe_orig, 1, 39).astype(np.float32)

if __name__ == '__main__':
    os.makedirs(save_npy_path, exist_ok=True)
    missing_files = _load_missing_file(missing_file_path)
    datalist = sorted([f for f in os.listdir(load_txt_path) if f.endswith('.skeleton')])
    
    print(f"📦 Step 1: NPY 변환 시작 (정규화 제외, 원본 스케일)")
    for ind, each in enumerate(datalist):
        _print_toolbar((ind + 1) / len(datalist), f"({ind+1}/{len(datalist)})")
        if each[:20] in missing_files: continue
        mat = _read_skeleton(os.path.join(load_txt_path, each))
        if mat is not None:
            np.save(os.path.join(save_npy_path, f"{each}.npy"), mat)
    print("\n✅ Step 1 완료!")