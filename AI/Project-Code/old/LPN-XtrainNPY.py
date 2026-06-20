import json
import numpy as np
import os

# coco의 json 파일을 따서 그 안에 있는 MoveNet과 같은 스켈레톤의 값을 사전에 설계한 13개의 좌표만 뽑는 작업.
# 하지만, 기존의 LPN과는 달리 가시성이 0(가려짐)인 상태는 무시하는 로직을 추가함. 0인데 정답이라고 배우면 안됨.
# 학습은 깨끗한 자료에서 하지만, 실전의 LPN은 가려진 데이터가 들어와도 주변의 데이터를 이용하여 정답을 추론함.
def preprocess_coco_for_lpn(json_path, save_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    print(f"Total annotations found: {len(annotations)}")

    # MoveNet(COCO) 17개 중 우리가 사용할 13개 인덱스 (0~16 기준)
    # 0:코, 5,6:어깨, 7,8:팔꿈치, 9,10:손목, 11,12:골반, 13,14:무릎, 15,16:발목
    # 눈(1,2)과 귀(3,4)는 제외합니다.
    target_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    processed_data = []

    for ann in annotations:
        # keypoints는 [x1, y1, v1, x2, y2, v2, ...] 형태 (v는 가시성)
        kp = ann.get('keypoints')
        
        # 데이터가 없거나 '사람'이 아니면 패스
        if not kp or ann.get('num_keypoints', 0) < 5:
            continue
            
        # 17개 관절을 [17, 3] 형태로 변환 (x, y, visibility)
        kp_array = np.array(kp).reshape(-1, 3)
        
        # 1. 우리가 정한 13개 관절만 추출
        selected_kp = kp_array[target_indices]
        
        # 2. 모든 관절이 찍혀있는 데이터만 사용 (v > 0)
        # LPN 학습 시 정답지가 명확해야 하므로 가려짐이 심한 데이터는 일단 배제
        if np.any(selected_kp[:, 2] == 0):
            continue
            
        # 3. 좌표 정규화 (Normalization)
        # 이미지 크기가 제각각이므로, 골반(여기선 11,12번의 중앙)을 0으로 잡거나 
        # 간단하게 해당 인물의 바운딩 박스 기준으로 정규화
        bbox = ann['bbox'] # [x, y, width, height]
        x_min, y_min, w, h = bbox
        
        # 2D 좌표 추출 (x, y만)
        coords_2d = selected_kp[:, :2]
        
        # 상대 좌표 변환 및 정규화
        coords_2d[:, 0] = (coords_2d[:, 0] - x_min) / w
        coords_2d[:, 1] = (coords_2d[:, 1] - y_min) / h
        
        # 1D 벡터로 펴기 (13 * 2 = 26차원)
        final_vector = coords_2d.flatten()
        processed_data.append(final_vector)

    # 1. float32로 데이터 타입 고정 (양자화 컨버터 호환성)
    result_npy = np.array(processed_data).astype(np.float32)
    np.save(save_path, result_npy)
    
    # 2. 양자화용 Representative Dataset 별도 저장 (약 500개 샘플)
    # 전체 데이터가 충분히 섞여(Shuffle) 있다고 가정하고 앞부분만 추출
    rep_data_path = save_path.replace('.npy', '_rep.npy')
    rep_samples = result_npy[:500] 
    np.save(rep_data_path, rep_samples)
    
    print(f"Final dataset shape: {result_npy.shape}")
    print(f"Saved to {save_path}")
    print(f"Representative dataset for Quantization saved to {rep_data_path}")

# 실행 (파일 경로를 본인 환경에 맞게 수정하세요)
preprocess_coco_for_lpn('./AI/dataset/coco-annotations/person_keypoints_train2017.json', './AI/dataset/LPN-train/coco_lpn_input_2d.npy')