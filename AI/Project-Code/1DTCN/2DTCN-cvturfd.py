import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ==========================================
# ⚙️ 1. 설정 및 경로 (오빠 환경에 맞게 수정)
# ==========================================
# GPU 메모리 효율적 할당 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 가속 활성화 완료")
    except RuntimeError as e:
        print(e)

urfd_base_dir = 'C:/Users/pcroom2/Desktop/ASK/AI/dataset/urfd'
save_dir = 'C:/Users/pcroom2/Desktop/ASK/AI/dataset/URFD_processed'

# 💡 [중요] saved_model.pb 파일이 들어있는 '폴더' 경로를 적어주세요.
MOVENET_TF2_PATH = 'C:/Users/pcroom2/Desktop/ASK/AI/models/movenet' 
LPN_MODEL_PATH = 'C:/Users/pcroom2/Desktop/ASK/AI/models/lpn/lpn_remaster_60_legacy.h5'

TARGET_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BATCH_SIZE = 32  # GPU 성능에 따라 16, 32, 64 조절 가능
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ==========================================
# 🧠 2. 모델 로딩
# ==========================================
print("🧠 MoveNet (SavedModel) 및 LPN 로딩 중...")
movenet = tf.saved_model.load(MOVENET_TF2_PATH)
movenet_fn = movenet.signatures['serving_default']

lpn_model = tf.keras.models.load_model(LPN_MODEL_PATH, compile=False)
print("✅ GPU 연산 준비 완료!")

# ==========================================
# ⚡ 3. 고속 배치 추론 함수
# ==========================================
def extract_poses_batch(img_paths):
    batch_res = []
    
    for path in img_paths:
        img = cv2.imread(path)
        if img is None: continue
        
        # 1. 이미지 전처리
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_rgb, (256, 256))
        
        # 2. 텐서 변환 및 배치 차원 추가 (1, 256, 256, 3)
        # MoveNet Thunder TF2는 배치 사이즈가 무조건 1이어야 함
        input_tensor = tf.convert_to_tensor([img_input], dtype=tf.int32)
        
        # 3. GPU 추론 (키워드 'input' 명시 필수)
        # 에러 메시지에서 요구한대로 input=... 형태로 전달
        outputs = movenet_fn(input=input_tensor)
        
        # 4. 결과값 추출 [1, 1, 17, 3] -> [17, 3]
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        res = []
        for idx in TARGET_INDICES:
            # 관절 좌표 [x, y] 순서로 저장 (오빠 로직 유지)
            res.extend([keypoints[idx][1], keypoints[idx][0]]) 
        batch_res.append(np.array(res))
        
    return batch_res

# ==========================================
# 🎬 4. 데이터 처리 로직
# ==========================================
def process_urfd_split(target_list):
    X_list, y_list = [], []
    
    for split in target_list:
        split_path = os.path.join(urfd_base_dir, split)
        if not os.path.exists(split_path): continue
            
        video_folders = sorted([f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))])
        
        for v_folder in tqdm(video_folders, desc=f"🎬 {split} 데이터 처리"):
            v_path = os.path.join(split_path, v_folder)
            label = 1 if 'fall' in v_folder.lower() else 0
            img_files = sorted(list(set(glob.glob(os.path.join(v_path, "*.png")) + glob.glob(os.path.join(v_path, "*.PNG")))))
            
            if len(img_files) < 10: continue

            # [배치 처리] 폴더 내 이미지를 배치 단위로 묶어서 GPU에 던짐
            pose_2d_seq = []
            for i in range(0, len(img_files), BATCH_SIZE):
                batch_files = img_files[i : i + BATCH_SIZE]
                poses = extract_poses_batch(batch_files)
                pose_2d_seq.extend(poses)
            
            if len(pose_2d_seq) < 5: continue
            
            # 60프레임 보간
            pose_2d_seq = np.array(pose_2d_seq)
            n_frames = len(pose_2d_seq)
            target_idx = np.linspace(0, n_frames - 1, 60)
            pose_2d_60 = np.zeros((60, 26), dtype=np.float32)
            for j in range(26):
                pose_2d_60[:, j] = np.interp(target_idx, np.arange(n_frames), pose_2d_seq[:, j])
            
            # LPN 3D 변환 (GPU 자동 연산)
            pose_3d_60 = lpn_model.predict(pose_2d_60.reshape(1, 60, 26), verbose=0)
            X_list.append(pose_3d_60[0])
            y_list.append(label)
            
    # 데이터 셔플
    X_array = np.array(X_list).astype(np.float32)
    y_array = np.array(y_list).reshape(-1, 1).astype(np.float32)
    
    if len(X_array) > 0:
        indices = np.arange(len(X_array))
        np.random.shuffle(indices)
        X_array, y_array = X_array[indices], y_array[indices]
    
    return X_array, y_array

# ==========================================
# 5. 실행 및 분리 저장
# ==========================================
if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    
    splits = {
        'train': ['train'],
        'val': ['vaild'],
        'test': ['test']
    }

    for name, folders in splits.items():
        X, y = process_urfd_split(folders)
        if len(X) > 0:
            np.save(f'{save_dir}/urfd_3d_{name}_X.npy', X)
            np.save(f'{save_dir}/urfd_3d_{name}_y.npy', y)
            print(f"💾 urfd_3d_{name} 저장 완료: {X.shape}")

    print("\n✅ 모든 작업이 빛의 속도로 완료되었습니다!")