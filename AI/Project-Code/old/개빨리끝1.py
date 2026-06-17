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
# 🚀 3. 파이프라인 엔진 (원본 전처리 및 Max Score 복구)
# ==========================================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    joint_buffer = []
    
    # [1단계] 영상 전체를 읽어 전처리된 좌표 추출
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img = cv2.resize(frame, (256, 256))
        img = img.reshape(1, 256, 256, 3).astype(np.uint8)
        mv_int.set_tensor(mv_in['index'], img)
        mv_int.invoke()
        kps_raw = mv_int.get_tensor(mv_out['index'])[0][0] 
        
        # 🔥 [핵심 복구] 사용자님의 원본 '프레임 단위' 전처리 로직
        target_2d = kps_raw[[0, 0, 5, 6, 7, 8, 9, 12, 11, 14, 13, 16, 15], :2]
        target_2d = target_2d[:, ::-1] 
        
        root_x = (target_2d[7, 0] + target_2d[8, 0]) / 2.0
        root_y = (target_2d[7, 1] + target_2d[8, 1]) / 2.0
        target_2d[:, 0] -= root_x
        target_2d[:, 1] -= root_y
        
        max_val = np.max(np.abs(target_2d)) + 1e-6
        target_2d /= max_val
        
        joint_buffer.append(target_2d.flatten())
        
    cap.release()
    
    raw_data = np.array(joint_buffer, dtype=np.float32)
    N = raw_data.shape[0]
    
    if N < WINDOW_SIZE: return None
    
    # [2단계] 추출된 전체 좌표를 STRIDE=2 윈도우로 분할 (전처리는 이미 끝남)
    windows = []
    for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
        windows.append(raw_data[start:start + WINDOW_SIZE])
    
    # [3단계] 분할된 윈도우들에 대해 모델 추론 및 Max Score 판정
    max_score = 0.0
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
        
        # 최고 낙상 확률 갱신
        if score > max_score:
            max_score = score
            
    # 🔥 영상 내 전체 구간 중 가장 높았던 확률이 0.5를 넘으면 낙상!
    video_pred = 1 if max_score > 0.5 else 0
    return video_pred

# ==========================================
# 📊 4. 실행부 (낙상 영상 10개 전용 테스트 모드)
# ==========================================
video_files = sorted(glob.glob(os.path.join(RGB_VIDEO_DIR, f"*{TEST_CAMERA}*.avi")))
y_true, y_pred = [], []

MAX_TEST_VIDEOS = 10  
processed_count = 0   

print(f"🚀 [Bug Fixed] 낙상(A043) 전용 성능 복구 검증 시작 (목표 샘플: {MAX_TEST_VIDEOS}개)")

for v_path in tqdm(video_files, desc="진행률"):
    filename = os.path.basename(v_path)
    action_id = int(filename.split('A')[1][:3])
    
    if action_id != 43: 
        continue

    label = 1 
    score = predict_video(v_path)
    
    if score is not None:
        y_true.append(label)
        y_pred.append(score)
        processed_count += 1
        
    if processed_count >= MAX_TEST_VIDEOS:
        tqdm.write(f"\n🛑 10개 영상 처리 완료.")
        break

# ==========================================
# 📈 5. 최종 리포트 출력 
# ==========================================
print("\n" + "="*50)
print(f"📊 1D-TCN Pipeline - 낙상 전용(A043) 평가 결과")
print("="*50)

if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"   총 영상 수: {len(y_true)} (전부 낙상)")
    print(f"   Accuracy:  {acc*100:.2f}%")
    print(f"   Recall (낙상 감지율): {rec*100:.2f}%") 
    print(f"\n   Confusion Matrix:")
    
    if cm.shape == (2, 2):
        print(f"                  Pred Normal  Pred Fall")
        print(f"   Actual Normal:   {cm[0][0]:>6}      {cm[0][1]:>6}")
        print(f"   Actual Fall:     {cm[1][0]:>6}      {cm[1][1]:>6}")
    elif cm.shape == (1, 1):
        fall_correct = np.sum(np.array(y_pred) == 1)
        fall_missed = np.sum(np.array(y_pred) == 0)
        print(f"                  Pred Normal  Pred Fall")
        print(f"   Actual Fall:     {fall_missed:>6}      {fall_correct:>6}")
else:
    print("❌ 처리된 영상이 없습니다.")
print("="*50)