import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# ⚙️ 1. 경로 및 모델 설정
# ==========================================
RGB_VIDEO_DIR = './AI/dataset/nturgb+d_rgb_C001/' 
MOVENET_MODEL = './AI/models/movenet/4.tflite' 
LN_MODEL      = './AI/models/lpn/lpn_cv_60_legacy_movenet_ptq.tflite'   
TCN_MODEL     = './AI/models/1d-tcn/tcn_8layer_combined_final_movenet.tflite'

WINDOW_SIZE = 60
STRIDE = 2
TEST_CAMERA = "C001" 

# ==========================================
# 🧠 2. 인터프리터 로드
# ==========================================
def load_pc_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return interpreter, in_det, out_det

mv_int, mv_in, mv_out = load_pc_model(MOVENET_MODEL)
ln_int, ln_in, ln_out = load_pc_model(LN_MODEL)
tcn_int, tcn_in, tcn_out = load_pc_model(TCN_MODEL)

ln_in_scale, ln_in_zero = ln_in['quantization']
ln_out_scale, ln_out_zero = ln_out['quantization']
tcn_in_scale, tcn_in_zero = tcn_in['quantization']
tcn_out_scale, tcn_out_zero = tcn_out['quantization']

# ==========================================
# 🛠️ 3. 전처리 유틸리티 (NPY 스크립트 로직 이식)
# ==========================================
def make_windows(skeleton_data):
    """(N, 26) 데이터를 STRIDE=2로 슬라이딩 윈도우 생성"""
    N = skeleton_data.shape[0]
    windows = []
    
    if N < WINDOW_SIZE:
        padded = np.zeros((WINDOW_SIZE, 26), dtype=np.float32)
        padded[:N] = skeleton_data
        windows.append(padded)
    else:
        for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
            windows.append(skeleton_data[start:start + WINDOW_SIZE])
            
    return np.array(windows, dtype=np.float32)

def process_windows(windows):
    """생성된 윈도우들에 대해 골반 센터링 및 스케일 정규화 일괄 수행"""
    # windows shape: (num_windows, 60, 26)
    reshaped = windows.reshape(-1, WINDOW_SIZE, 13, 2)
    
    # 7:L-Hip, 8:R-Hip의 중앙을 Root로 설정
    root = (reshaped[:, :, 7:8, :] + reshaped[:, :, 8:9, :]) / 2.0
    centered = reshaped - root
    
    # 각 윈도우별 정규화 (Z축 비율 보존 방식 적용)
    max_vals = np.max(np.abs(centered), axis=(1,2,3), keepdims=True) + 1e-6
    normed = centered / max_vals
    
    return normed.reshape(-1, WINDOW_SIZE, 26)

# ==========================================
# 🚀 4. 파이프라인 엔진 (과반수 투표 적용)
# ==========================================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    raw_skeletons = []
    
    # [1단계] 영상 전체를 읽어 MoveNet 좌표만 빠르게 추출 (I/O 루프 분리)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img = cv2.resize(frame, (256, 256))
        img = img.reshape(1, 256, 256, 3).astype(np.uint8)
        mv_int.set_tensor(mv_in['index'], img)
        mv_int.invoke()
        kps_raw = mv_int.get_tensor(mv_out['index'])[0][0] 
        
        # 13개 관절 매핑 및 [x, y] 반전
        target_2d = kps_raw[[0, 0, 5, 6, 7, 8, 9, 12, 11, 14, 13, 16, 15], :2]
        target_2d = target_2d[:, ::-1] 
        
        raw_skeletons.append(target_2d.flatten())
        
    cap.release()
    
    if len(raw_skeletons) == 0: return None
    
    # [2단계] 추출된 전체 좌표를 STRIDE=2 윈도우로 분할 및 전처리
    raw_data = np.array(raw_skeletons)
    windows = make_windows(raw_data)
    windows = process_windows(windows)
    
    # [3단계] 분할된 윈도우들에 대해 모델 추론 및 과반수 투표
    preds = []
    for seq_2d in windows:
        seq_2d = seq_2d.reshape(1, 60, 26)
        
        # LN 추론
        seq_2d_q = np.clip(np.round(seq_2d / ln_in_scale + ln_in_zero), -128, 127).astype(np.int8)
        ln_int.set_tensor(ln_in['index'], seq_2d_q)
        ln_int.invoke()
        ln_out_f = (ln_int.get_tensor(ln_out['index']).astype(np.float32) - ln_out_zero) * ln_out_scale
        
        # TCN 추론
        tcn_in_q = np.clip(np.round(ln_out_f / tcn_in_scale + tcn_in_zero), -128, 127).astype(np.int8)
        tcn_int.set_tensor(tcn_in['index'], tcn_in_q)
        tcn_int.invoke()
        res_f = (tcn_int.get_tensor(tcn_out['index']).astype(np.float32) - tcn_out_zero) * tcn_out_scale
        
        score = res_f.flatten()[1] if res_f.size > 1 else res_f.flatten()[0]
        preds.append(1 if score > 0.5 else 0)
        
    # Majority Vote: 과반수가 낙상이면 해당 영상 = 낙상
    video_pred = 1 if np.mean(preds) > 0.5 else 0
    return video_pred

# ==========================================
# 📊 5. 실행부
# ==========================================
video_files = sorted(glob.glob(os.path.join(RGB_VIDEO_DIR, f"*{TEST_CAMERA}*.avi")))
y_true, y_pred = [], []

print(f"🚀 [Video-Level Majority Vote] 쾌속 모드 검증 시작 (샘플: {len(video_files)})")

for v_path in tqdm(video_files, desc="전체 비디오 진행률"):
    filename = os.path.basename(v_path)
    action_id = int(filename.split('A')[1][:3])
    
    # 2인 상호작용 액션 평가에서 제외
    if (50 <= action_id <= 60) or (106 <= action_id <= 120): 
        continue

    label = 1 if action_id == 43 else 0
    score = predict_video(v_path)
    
    if score is not None:
        y_true.append(label)
        y_pred.append(score)

# ==========================================
# 📈 6. 최종 리포트 출력 (NPY 스크립트와 동일한 포맷)
# ==========================================
print("\n" + "="*50)
print(f"📊 1D-TCN Pipeline - Video-Level 평가 결과")
print("="*50)

if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"   총 영상 수: {len(y_true)} (낙상: {np.sum(np.array(y_true)==1)}, 비낙상: {np.sum(np.array(y_true)==0)})")
    print(f"   Accuracy:  {acc*100:.2f}%")
    print(f"   Precision: {pre*100:.2f}%")
    print(f"   Recall:    {rec*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    print(f"\n   Confusion Matrix:")
    print(f"                  Pred Normal  Pred Fall")
    print(f"   Actual Normal:   {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"   Actual Fall:     {cm[1][0]:>6}      {cm[1][1]:>6}")
else:
    print("❌ 처리된 영상이 없습니다.")
print("="*50)