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
TCN_H5_PATH = './AI/models/1d-tcn/tcn_8layer_pure_ntu.h5'
TCN_TFLITE_PATH = './AI/models/1d-tcn/tcn_manual_fixed.tflite'

TEST_CAMERA = "C001" 
TARGET_FRAMES = 60

# ==========================================
# 🛠️ 2. 전처리 로직 (성공했던 옛날 방식)
# ==========================================
def preprocess_old_success_logic(skeleton_data):
    if len(skeleton_data.shape) == 3:
        skeleton_data = np.squeeze(skeleton_data, axis=1)
    
    nframe_orig = skeleton_data.shape[0]
    if nframe_orig == 0: return None

    # 1. 60프레임 선형 보간 (Linear Interpolation)
    orig_indices = np.arange(nframe_orig)
    new_indices = np.linspace(0, nframe_orig - 1, TARGET_FRAMES)
    
    tmp_reshaped = skeleton_data.reshape(nframe_orig, 13, 3)
    skel_60 = np.zeros((TARGET_FRAMES, 13, 3), dtype=np.float32)
    
    for j in range(13):
        for c in range(3):
            skel_60[:, j, c] = np.interp(new_indices, orig_indices, tmp_reshaped[:, j, c])

    # 2. 중심점 이동 (골반 기준)
    root = (skel_60[:, 7:8, :] + skel_60[:, 8:9, :]) / 2
    skel_relative = skel_60 - root

    # 3. max_dist 정규화 (양자화 열화 방지의 핵심)
    max_dist = np.max(np.abs(skel_relative)) + 1e-6
    skel_norm = skel_relative / max_dist

    return skel_norm.reshape(1, TARGET_FRAMES, 39).astype(np.float32)

# ==========================================
# 🧠 3. 통합 평가 루프
# ==========================================
def evaluate_models(skeleton_files):
    model_h5 = tf.keras.models.load_model(TCN_H5_PATH, compile=False)
    
    interpreter = tf.lite.Interpreter(model_path=TCN_TFLITE_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zero = in_det['quantization']
    out_scale, out_zero = out_det['quantization']

    y_true, y_pred_h5, y_pred_tflite = [], [], []

    for fpath in tqdm(skeleton_files, desc="Evaluating"):
        filename = os.path.basename(fpath)
        label = 1 if "A043" in filename else 0
        
        try:
            raw = np.load(fpath)
            inp = preprocess_old_success_logic(raw)
            if inp is None: continue
        except: continue

        # --- H5 추론 ---
        prob_h5 = model_h5.predict(inp, verbose=0)
        y_pred_h5.append(1 if prob_h5.flatten()[0] > 0.5 else 0)

        # --- TFLite 추론 ---
        inp_q = np.clip(np.round(inp / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(in_det['index'], inp_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(out_det['index'])
        prob_tflite = (out_q.astype(np.float32) - out_zero) * out_scale
        y_pred_tflite.append(1 if prob_tflite.flatten()[0] > 0.5 else 0)

        y_true.append(label)

    return np.array(y_true), np.array(y_pred_h5), np.array(y_pred_tflite)

# ==========================================
# 📊 4. 결과 출력 (4대 지표)
# ==========================================
def print_metrics(tag, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 [{tag}] 평가 결과")
    print(f"{'='*50}")
    print(f"✅ Accuracy  : {acc*100:.2f}%")
    print(f"✅ Precision : {pre*100:.2f}%")
    print(f"✅ Recall    : {rec*100:.2f}%")
    print(f"✅ F1-Score  : {f1*100:.2f}%")
    print(f"\n[Confusion Matrix]\n{cm}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    all_files = sorted(glob.glob(os.path.join(NTU_SKELETON_DIR, "*.skeleton.npy")))
    test_files = [f for f in all_files if TEST_CAMERA in os.path.basename(f)]
    
    y_true, y_h5, y_tflite = evaluate_models(test_files)
    
    print_metrics("1D-TCN H5 (FP32)", y_true, y_h5)
    print_metrics("1D-TCN TFLite (INT8)", y_true, y_tflite)