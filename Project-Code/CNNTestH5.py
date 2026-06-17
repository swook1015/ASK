"""
영상 단위(Video-Level) 평가 스크립트 (이중 센터링 제거 버전)
학습 데이터(Step 2)와 동일하게 윈도우 단위 정규화만 적용하여 평가 수행
"""
import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# ⚙️ 1. 설정
# ==========================================
# Step 1의 결과물(이미 센터링 완료된 npy)이 저장된 폴더
NTU_SKELETON_DIR = './AI/dataset/nturgb+d_skeletons-npy/'
TCN_H5_PATH = './AI/models/1d-tcn/tcn_8layer_pure_ntu.h5'
TCN_TFLITE_PATH = './AI/models/1d-tcn/tcn_manual_fixed.tflite'

TEST_CAMERA = "C001"       # Cross-View Test: 카메라 1 (-45°)
WINDOW_SIZE = 60
STRIDE = 2

# ==========================================
# 🛠️ 2. 윈도우 생성 및 정규화 (학습 환경 일치)
# ==========================================
# 🛠️ 2. 윈도우 생성 및 정규화 (학습 환경 완벽 일치 버전)
def make_windows_and_normalize(skeleton_data):
    """
    원본 스켈레톤 -> 슬라이딩 윈도우 생성 및 정규화 (cv-npy2.py 완벽 호환)
    """
    N = skeleton_data.shape[0]
    windows = []
    
    # 차원 정리: (N, 1, 39) -> (N, 39)
    if len(skeleton_data.shape) == 3:
        skeleton_data = np.squeeze(skeleton_data, axis=1).copy()
    else:
        skeleton_data = skeleton_data.copy()

    # 🚨 [핵심 1] 학습 데이터와 동일하게 Y축 부호 반전 (인덱스 1, 4, 7...)
    skeleton_data[:, 1::3] = -skeleton_data[:, 1::3]

    # 🚨 [핵심 2] 동적 max_dist 제거하고 학습 때 썼던 480.0 고정 상수 적용
    STATIC_SCALE = 480.0

    if N < WINDOW_SIZE:
        window = np.zeros((WINDOW_SIZE, 39), dtype=np.float32)
        window[:N] = skeleton_data
        windows.append(window / STATIC_SCALE)
    else:
        for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
            window = skeleton_data[start:start + WINDOW_SIZE]
            windows.append(window / STATIC_SCALE)
    
    return np.array(windows, dtype=np.float32)

# ==========================================
# 🧠 3. 영상 단위 평가 (H5)
# ==========================================
def evaluate_video_level_h5(model, skeleton_files):
    y_true_video = []
    y_pred_video = []
    
    for fpath in tqdm(skeleton_files, desc="H5 Video-Level"):
        filename = os.path.basename(fpath)
        label = 1 if "A043" in filename else 0
        
        try:
            raw = np.load(fpath) # Step 1에서 이미 센터링된 데이터
            if raw.shape[0] == 0: continue
        except:
            continue
        
        # 윈도우 생성 및 정규화 (center_poses 제거됨)
        windows = make_windows_and_normalize(raw)
        
        # 추론
        probs = model.predict(windows, batch_size=256, verbose=0)
        preds = (probs > 0.5).astype(int).flatten()
        
        # Majority Vote (과반수 투표)
        video_pred = 1 if np.mean(preds) > 0.5 else 0
        
        y_true_video.append(label)
        y_pred_video.append(video_pred)
    
    return np.array(y_true_video), np.array(y_pred_video)

# ==========================================
# 🧠 4. 영상 단위 평가 (TFLite)
# ==========================================
def evaluate_video_level_tflite(tflite_path, skeleton_files):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zero = in_det['quantization']
    out_scale, out_zero = out_det['quantization']
    
    y_true_video = []
    y_pred_video = []
    
    for fpath in tqdm(skeleton_files, desc="TFLite Video-Level"):
        filename = os.path.basename(fpath)
        label = 1 if "A043" in filename else 0
        
        try:
            raw = np.load(fpath)
            if raw.shape[0] == 0: continue
        except:
            continue
        
        # 윈도우 생성 및 정규화 (center_poses 제거됨)
        windows = make_windows_and_normalize(raw)
        
        # TFLite 추론
        preds = []
        for i in range(len(windows)):
            inp = windows[i:i+1] # (1, 60, 39)
            inp_q = np.clip(np.round(inp / in_scale + in_zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(in_det['index'], inp_q)
            interpreter.invoke()
            out_q = interpreter.get_tensor(out_det['index'])
            out_f = (out_q.astype(np.float32) - out_zero) * out_scale
            preds.append(1 if out_f > 0.5 else 0)
        
        # Majority Vote
        video_pred = 1 if np.mean(preds) > 0.5 else 0
        
        y_true_video.append(label)
        y_pred_video.append(video_pred)
    
    return np.array(y_true_video), np.array(y_pred_video)

# ==========================================
# 📊 5. 결과 출력 함수 (동일)
# ==========================================
def print_results(tag, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}\n📊 {tag} - Video-Level 결과\n{'='*50}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {pre*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"\n  Confusion Matrix:\n{cm}\n{'='*50}")

# ==========================================
# 🚀 6. 실행
# ==========================================
if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(NTU_SKELETON_DIR, "*.skeleton.npy")))
    test_files = [f for f in all_files if TEST_CAMERA in os.path.basename(f)]
    
    print(f"📦 NTU Cross-View Test 시작 (C001 대상, {len(test_files)}개)")
    
    # --- H5 평가 ---
    print("\n🚀 H5 모델 평가...")
    model_h5 = tf.keras.models.load_model(TCN_H5_PATH, compile=False)
    y_true_h5, y_pred_h5 = evaluate_video_level_h5(model_h5, test_files)
    print_results("1D-TCN H5 (FP32)", y_true_h5, y_pred_h5)
    
    # --- TFLite 평가 ---
    print("\n🚀 TFLite 모델 평가...")
    y_true_q, y_pred_q = evaluate_video_level_tflite(TCN_TFLITE_PATH, test_files)
    print_results("1D-TCN TFLite (INT8)", y_true_q, y_pred_q)