import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

# 💡 1. 경로 설정
H5_PATH = './AI/models/ln/0408new_lpn_cv_60_legacy.h5' 
TFLITE_PATH = './AI/models/ln/0408new_lpn_cv_60_legacy.tflite'
X_TEST_PATH = './AI/dataset/CV_Split/X_cv_test.npy'

# ---------------------------------------------------------
# 🛠️ [Keras 3 패치] 레이어 호환성 문제 해결 클래스
# ---------------------------------------------------------
class PatchedBatchNormalization(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        if 'axis' in config and isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return super(PatchedBatchNormalization, cls).from_config(config)

# 💡 2. 데이터 로드 함수
def load_lpn_data(path):
    print(f"📦 테스트 데이터 로딩 중: {os.path.basename(path)}")
    raw_data = np.load(path).astype(np.float32)
    N, T, _ = raw_data.shape
    Y_true = raw_data
    data_reshaped = raw_data.reshape(N, T, 13, 3)
    X_input = data_reshaped[:, :, :, 0:2].reshape(N, T, 26)
    return X_input, Y_true

# 💡 3. MPJPE 계산 함수
def calc_mpjpe_mm(y_true, y_pred):
    y_true_3d = y_true.reshape(-1, 60, 13, 3)
    y_pred_3d = y_pred.reshape(-1, 60, 13, 3)
    dist = np.sqrt(np.sum((y_true_3d - y_pred_3d)**2, axis=-1))
    return np.mean(dist) * 1000

# 🚀 메인 실행부
if __name__ == "__main__":
    if not os.path.exists(H5_PATH):
        print(f"❌ H5 파일을 찾을 수 없습니다: {H5_PATH}")
        exit()

    X_test, Y_test = load_lpn_data(X_TEST_PATH)
    N = len(X_test)

    # ---------------------------------------------------------
    # 🧠 [H5 로드]
    # ---------------------------------------------------------
    print("\n📦 H5 모델 로드 중 (Keras 3 통합 패치 모드)...")
    custom_objs = {
        'Dense': tf.keras.layers.Dense,
        'Activation': tf.keras.layers.Activation,
        'BatchNormalization': PatchedBatchNormalization,
        'Dropout': tf.keras.layers.Dropout,
        'ReLU': tf.keras.layers.ReLU,
        'TimeDistributed': tf.keras.layers.TimeDistributed,
        'InputLayer': tf.keras.layers.InputLayer,
        'Add': tf.keras.layers.Add,
        'Multiply': tf.keras.layers.Multiply,
        'Reshape': tf.keras.layers.Reshape,
        'Concatenate': tf.keras.layers.Concatenate
    }

    try:
        model_h5 = tf.keras.models.load_model(H5_PATH, compile=False, custom_objects=custom_objs)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit()

    # ---------------------------------------------------------
    # 🧠 [TFLite 로드]
    # ---------------------------------------------------------
    print("📦 TFLite 모델 로드 중...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zero = in_det['quantization']
    out_scale, out_zero = out_det['quantization']

    # ---------------------------------------------------------
    # ⚡ [초고속 수정 파트] 4. 분리된 추론 로직
    # ---------------------------------------------------------
    
    # 1. H5 모델은 for문 밖에서 '배치(Batch) 단위'로 한 번에 밀어넣습니다. (초고속)
    print("\n🚀 [H5] 전체 데이터 추론 시작 (배치 처리)...")
    Y_pred_h5 = model_h5.predict(X_test, batch_size=1024, verbose=1)

    # 2. TFLite는 입력 규격이 고정되어 있어 어쩔 수 없이 1개씩 쪼개 넣습니다.
    y_pred_tflite = np.zeros((N, 60, 39), dtype=np.float32)
    print(f"\n🚀 [TFLite] 추론 시작 (총 {N}개 시퀀스)...")

    for i in tqdm(range(N), desc="TFLite Processing"):
        inp = X_test[i:i+1]
        
        # 양자화 (INT8)
        if in_scale != 0:
            inp_q = np.clip(np.round(inp / in_scale + in_zero), -128, 127).astype(np.int8)
        else:
            inp_q = inp
            
        interpreter.set_tensor(in_det['index'], inp_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(out_det['index'])
        
        # 역양자화 (FP32)
        if out_scale != 0:
            out_f = (out_q.astype(np.float32) - out_zero) * out_scale
        else:
            out_f = out_q
            
        y_pred_tflite[i] = out_f

    Y_pred_tflite = y_pred_tflite

    # ---------------------------------------------------------
    # 💡 5. 최종 결과 출력
    # ---------------------------------------------------------
    mpjpe_h5 = calc_mpjpe_mm(Y_test, Y_pred_h5)
    mpjpe_tflite = calc_mpjpe_mm(Y_test, Y_pred_tflite)

    print("\n" + "="*60)
    print(f"📊 [LPN 모델 H5 vs TFLite 전수 검증 결과]")
    print(f"{'='*60}")
    print(f"✅ FP32 (H5) MPJPE     : {mpjpe_h5:.2f} mm")
    print(f"✅ INT8 (TFLite) MPJPE   : {mpjpe_tflite:.2f} mm")
    print(f"✅ 성능 열화(Degradation) : {mpjpe_tflite - mpjpe_h5:.2f} mm")
    print("="*60)