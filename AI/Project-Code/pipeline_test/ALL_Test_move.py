import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# ⚙️ 1. 설정
# ==========================================
NTU_SKELETON_DIR = './AI/dataset/nturgb+d_skeletons-npy/'
LN_MODEL      = './AI/models/lpn/lpn_cv_60_legacy_movenet_ptq.tflite'   
TCN_MODEL     = './AI/models/1d-tcn/tcn_8layer_combined_final_movenet.tflite'

TEST_CAMERA = "C001"       
WINDOW_SIZE = 60
STRIDE = 2

# ==========================================
# 🧠 2. 인터프리터 로드
# ==========================================
def load_pc_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return interpreter, in_det, out_det

ln_int, ln_in, ln_out = load_pc_model(LN_MODEL)
tcn_int, tcn_in, tcn_out = load_pc_model(TCN_MODEL)

ln_in_scale, ln_in_zero = ln_in['quantization']
ln_out_scale, ln_out_zero = ln_out['quantization']
tcn_in_scale, tcn_in_zero = tcn_in['quantization']
tcn_out_scale, tcn_out_zero = tcn_out['quantization']

# ==========================================
# 🛠️ 3. 전처리 유틸리티 (🔥 행렬 벡터화로 초고속 처리)
# ==========================================
def preprocess_to_2d_fast(raw_3d_data):
    """(N, 39) -> (N, 26) 변환 및 정규화를 for문 없이 한방에 처리"""
    N = raw_3d_data.shape[0]
    # 3D -> 2D (x, y만 추출)
    raw_2d = raw_3d_data.reshape(N, 13, 3)[:, :, :2] 
    
    # 골반(7, 8번) 중심 계산 (N, 1, 2)
    root = (raw_2d[:, 7:8, :] + raw_2d[:, 8:9, :]) / 2.0
    
    # 센터링
    centered = raw_2d - root
    
    # 프레임별 최대값으로 정규화
    max_vals = np.max(np.abs(centered), axis=(1,2), keepdims=True) + 1e-6
    normed = centered / max_vals
    
    return normed.reshape(N, 26).astype(np.float32)

def make_windows(skeleton_2d_data):
    N = skeleton_2d_data.shape[0]
    windows = []
    if N < WINDOW_SIZE:
        padded = np.zeros((WINDOW_SIZE, 26), dtype=np.float32)
        padded[:N] = skeleton_2d_data
        windows.append(padded)
    else:
        for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
            windows.append(skeleton_2d_data[start:start + WINDOW_SIZE])
    return np.array(windows, dtype=np.float32)

# ==========================================
# 🚀 4. 파이프라인 엔진
# ==========================================
def evaluate_pipeline(skeleton_files):
    y_true, y_pred = [], []
    
    for fpath in tqdm(skeleton_files, desc="전체 비디오 진행률"):
        filename = os.path.basename(fpath)
        action_id = int(filename.split('A')[1][:3])
        
        if (50 <= action_id <= 60) or (106 <= action_id <= 120): 
            continue
            
        label = 1 if action_id == 43 else 0
        
        try:
            raw = np.load(fpath)
            if len(raw.shape) == 3: raw = np.squeeze(raw, axis=1)
            if raw.shape[0] == 0: continue
        except:
            continue
            
        # 🔥 고속 전처리
        data_2d = preprocess_to_2d_fast(raw)
        windows = make_windows(data_2d)
        
        max_score = 0.0 # 🔥 치명적 오류였던 과반수 투표 대신 Max Score 적용
        
        for i in range(len(windows)):
            seq_2d = windows[i:i+1] # (1, 60, 26)
            
            # LN
            seq_2d_q = np.clip(np.round(seq_2d / ln_in_scale + ln_in_zero), -128, 127).astype(np.int8)
            ln_int.set_tensor(ln_in['index'], seq_2d_q)
            ln_int.invoke()
            ln_out_f = (ln_int.get_tensor(ln_out['index']).astype(np.float32) - ln_out_zero) * ln_out_scale
            
            # TCN
            tcn_in_q = np.clip(np.round(ln_out_f / tcn_in_scale + tcn_in_zero), -128, 127).astype(np.int8)
            tcn_int.set_tensor(tcn_in['index'], tcn_in_q)
            tcn_int.invoke()
            res_f = (tcn_int.get_tensor(tcn_out['index']).astype(np.float32) - tcn_out_zero) * tcn_out_scale
            
            score = res_f.flatten()[1] if res_f.size > 1 else res_f.flatten()[0]
            if score > max_score:
                max_score = score
                
        # 윈도우 중 단 하나라도 낙상 임계치를 넘으면 낙상 판정
        video_pred = 1 if max_score > 0.5 else 0
        
        y_true.append(label)
        y_pred.append(video_pred)
        
    return np.array(y_true), np.array(y_pred)

# ==========================================
# 📊 5. 실행부
# ==========================================
if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(NTU_SKELETON_DIR, "*.skeleton.npy")))
    test_files = [f for f in all_files if TEST_CAMERA in os.path.basename(f)]
    
    print(f"📦 NTU Cross-View Test 로드 (샘플: {len(test_files)}개)")
    print("\n🚀 [Fast Pipeline] NPY 고속 처리 및 Max Score 평가 시작...")
    
    y_true, y_pred = evaluate_pipeline(test_files)
    
    print("\n" + "="*50)
    print(f"📊 Full Pipeline (LN + 1D-TCN) - 최종 성능표")
    print("="*50)

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"   총 평가 영상 수: {len(y_true)}")
        print(f"   Accuracy:  {acc*100:.2f}%")
        print(f"   Precision: {pre*100:.2f}%")
        print(f"   Recall:    {rec*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        print(f"\n   Confusion Matrix:")
        print(f"                  Pred Normal  Pred Fall")
        print(f"   Actual Normal:   {cm[0][0]:>6}      {cm[0][1]:>6}")
        print(f"   Actual Fall:     {cm[1][0]:>6}      {cm[1][1]:>6}")
    else:
        print("❌ 유효한 영상 데이터가 없습니다.")
    print("="*50)