import cv2
import numpy as np
import os
import tensorflow as tf

# ==========================================
# ⚙️ 1. 경로 및 설정
# ==========================================
POSE_MODEL = 'mspt.tflite'  # PC용 MoveNet 모델 파일명 (경로 확인 필수)
TRAINSET_DIR = './trainset' # URFD 이미지 폴더들이 모인 상위 디렉토리
WINDOW_SIZE = 20

# 🔥 [핵심] MoveNet의 17개 관절 중 NTU 기초학습과 동일한 13개만 추출!
# (코, 왼쪽/오른쪽 어깨~발목) -> 눈과 귀는 버림
MOVENET_13_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# MoveNet TFLite 인터프리터 설정
interpreter = tf.lite.Interpreter(model_path=POSE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]

def process_and_save_urfd():
    all_X, all_y = [], []
    
    # 폴더 목록 불러오기 (정렬 필수)
    folders = sorted(os.listdir(TRAINSET_DIR))
    
    for folder_name in folders:
        folder_path = os.path.join(TRAINSET_DIR, folder_name)
        if not os.path.isdir(folder_path): continue
        
        # 🏷️ 폴더명으로 라벨 결정 ('fall'이 포함되면 낙상 1, 아니면 일상 0)
        label = 1 if 'fall' in folder_name.lower() else 0
        
        # 🖼️ 이미지 파일 수집 및 정렬
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(root, f))
        
        image_paths.sort() # 시간 순서대로 정렬 (매우 중요)
        
        if len(image_paths) < WINDOW_SIZE:
            print(f"⚠️ {folder_name}: 프레임이 {WINDOW_SIZE}장 미만이라 스킵합니다.")
            continue

        print(f"📂 관절 추출 중: {folder_name} (총 {len(image_paths)}장)")
        folder_keypoints = []
        
        # 1️⃣ 이미지마다 MoveNet을 돌려서 13개 관절 좌표만 뽑기
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # MoveNet 입력 크기에 맞게 리사이즈
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (w, h))
            input_data = np.expand_dims(img, axis=0).astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # 출력 데이터 (1, 1, 17, 3)에서 (17, 3) 추출
            raw_keypoints = interpreter.get_tensor(output_details[0]['index'])[0, 0]
            
            # 🔥 13개 관절 선택 및 X, Y 좌표만 가져오기 (Score 제외) -> (13, 2)
            selected_kp = raw_keypoints[MOVENET_13_INDICES, :2]
            
            # 1D-CNN 입력용으로 1차원으로 쫙 펴기 -> 길이가 26이 됨
            folder_keypoints.append(selected_kp.flatten())
            
        # 2️⃣ 20프레임씩 슬라이딩 윈도우로 묶어주기 (데이터 증강 효과)
        for i in range(len(folder_keypoints) - WINDOW_SIZE + 1):
            seq = np.array(folder_keypoints[i : i + WINDOW_SIZE]) # (20, 26)
            
            # 🔥 [가장 중요] NTU 기초학습 때 썼던 동일한 방식의 정규화 (Min-Max)
            denom = np.max(seq) - np.min(seq) + 1e-6
            seq_norm = (seq - np.min(seq)) / denom
            
            all_X.append(seq_norm)
            all_y.append(label)

    # 3️⃣ 데이터 셔플 및 최종 저장
    final_X = np.array(all_X)
    final_y = np.array(all_y)
    
    # 🔀 [핵심 추가] 데이터 쌍을 유지하면서 랜덤하게 섞기
    print("\n🔀 추출된 데이터를 골고루 섞는 중 (Shuffling)...")
    indices = np.arange(final_X.shape[0])
    np.random.shuffle(indices)
    
    final_X = final_X[indices]
    final_y = final_y[indices]
    
    np.save('X_urfd.npy', final_X)
    np.save('y_urfd.npy', final_y)
    
    print("\n" + "="*50)
    print(f"✅ URFD 데이터 셔플 및 저장 완료!")
    print(f" - X_urfd.npy 형태: {final_X.shape} (샘플 수, 20프레임, 26특징)")
    print(f" - y_urfd.npy 형태: {final_y.shape}")
    print("="*50)

if __name__ == "__main__":
    process_and_save_urfd()