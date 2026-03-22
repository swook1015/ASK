import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. 모델 및 데이터 로드 (오빠 경로에 맞게 수정)
MODEL_PATH = './AI/models/1d-tcn/tcn_fall_detector_stride_best.h5'
X_test = np.load('./combined_train_X.npy').astype(np.float32)
y_true = np.load('./combined_train_y.npy').astype(np.int32)

model = load_model(MODEL_PATH)
THRESHOLD = 0.85

# 2. 전체 데이터 추론 (속도를 위해 배치 처리)
print("🚀 전체 데이터 추론 시작...")
y_probs = model.predict(X_test, batch_size=256, verbose=1).flatten()
y_pred = (y_probs > THRESHOLD).astype(np.int32)

# 3. 수치 성능 계산
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "="*30)
print(f"✅ 모델 성능 요약 (Threshold: {THRESHOLD})")
print(f"- 정확도 (Accuracy):  {acc:.4f}")
print(f"- 정밀도 (Precision): {prec:.4f} (낙상이라 한 것 중 진짜 낙상)")
print(f"- 재현율 (Recall):    {rec:.4f} (진짜 낙상 중 찾아낸 비율)")
print(f"- F1-Score:           {f1:.4f}")
print("="*30)

# 4. 🔥 오답 노트 (Error Analysis) 추출
# 미탐지 (False Negative): 실제 낙상(1)인데 정상(0)으로 판단
fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
# 오탐지 (False Positive): 실제 정상(0)인데 낙상(1)으로 판단
fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

print(f"\n📝 [오답 노트] 분석 결과")
print(f"1. 미탐지(놓친 낙상): 총 {len(fn_indices)}건")
print(f"   -> 주요 인덱스: {fn_indices[:20]} ...") # 상위 20개만 출력

print(f"2. 오탐지(가짜 알람): 총 {len(fp_indices)}건")
print(f"   -> 주요 인덱스: {fp_indices[:20]} ...")

# 5. 오답 리스트 저장 (나중에 데이터 보강용)
np.savetxt('false_negative_indices.txt', fn_indices, fmt='%d')
np.savetxt('false_positive_indices.txt', fp_indices, fmt='%d')
print("\n💾 오답 인덱스가 .txt 파일로 저장되었습니다.")