import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 💡 Keras 2 Legacy 설정 (필요시)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 1. 경로 설정
model_path = './AI/models/lpn/lpn_cv_60_legacy.h5'
input_2d_path = './AI/dataset/CV-Split/X_cv_test.npy'
target_3d_path = './AI/dataset/CV-Split/y_cv_test.npy'

# 2. 모델 및 데이터 로드
model = tf.keras.models.load_model(model_path)
data_2d = np.load(input_2d_path).astype('float32')   # (N, 26)
data_3d_gt = np.load(target_3d_path).astype('float32') # (N, 39)

# 💡 모델이 요구하는 시퀀스 길이 설정
SEQ_LEN = 60

# 3. 뼈대 연결
EDGES = [
    (0, 1), (0, 2), (1, 2),   # 상체
    (1, 3), (3, 5),           # 왼팔
    (2, 4), (4, 6),           # 오른팔
    (1, 7), (2, 8), (7, 8),   # 몸통
    (7, 9), (9, 11),          # 왼다리
    (8, 10), (10, 12)         # 오른다리
]

# 💡 시작 인덱스를 60프레임 이후로 설정 (과거 데이터가 필요하므로)
current_idx = SEQ_LEN 

def draw_comparison(idx):
    plt.clf()
    
    # 🚨 [핵심 수정] 1프레임이 아니라 과거 60프레임을 묶어서 추출
    # 결과 Shape: (60, 26) -> (1, 60, 26)
    sample_seq = data_2d[idx-SEQ_LEN:idx].reshape(1, SEQ_LEN, 26)
    
    # 예측 수행 -> 결과 Shape: (1, 60, 39)
    full_pred = model.predict(sample_seq, verbose=0)
    
    # 💡 60개 결과 중 가장 마지막(현재 프레임) 결과만 시각화
    pred_3d = full_pred[0, -1].reshape(13, 3)
    gt_3d = data_3d_gt[idx-1].reshape(13, 3)
    in_2d = data_2d[idx-1].reshape(13, 2)

    # --- 1. 왼쪽: Input 2D ---
    ax1 = fig.add_subplot(131)
    ax1.scatter(in_2d[:, 0], in_2d[:, 1], c='blue', s=20)
    for e in EDGES:
        ax1.plot([in_2d[e[0],0], in_2d[e[1],0]], [in_2d[e[0],1], in_2d[e[1],1]], c='gray', alpha=0.5)
    ax1.set_title(f"Input 2D (Frame {idx})")
    ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([1.2, -1.2])

    # --- 2. 가운데: Predicted 3D (LPN) ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c='red', s=20)
    for e in EDGES:
        ax2.plot([pred_3d[e[0],0], pred_3d[e[1],0]], [pred_3d[e[0],1], pred_3d[e[1],1]], [pred_3d[e[0],2], pred_3d[e[1],2]], c='red')
    ax2.set_title("LPN Predicted 3D")
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    ax2.view_init(elev=20, azim=30)

    # --- 3. 오른쪽: Ground Truth 3D ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c='green', s=20)
    for e in EDGES:
        ax3.plot([gt_3d[e[0],0], gt_3d[e[1],0]], [gt_3d[e[0],1], gt_3d[e[1],1]], [gt_3d[e[0],2], gt_3d[e[1],2]], c='green')
    ax3.set_title("Ground Truth 3D")
    ax3.set_xlim([-1, 1]); ax3.set_ylim([-1, 1]); ax3.set_zlim([-1, 1])
    ax3.view_init(elev=20, azim=30)

    plt.tight_layout()
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == ' ':
        # 랜덤 샘플링 시에도 SEQ_LEN 이상의 인덱스 선택
        current_idx = np.random.randint(SEQ_LEN, len(data_2d))
        draw_comparison(current_idx)
    elif event.key == 'escape':
        plt.close()

fig = plt.figure(figsize=(18, 6))
fig.canvas.mpl_connect('key_press_event', on_key)

print("🎹 [Space]: 랜덤 시퀀스 샘플 비교 | [ESC]: 종료")
draw_comparison(current_idx)
plt.show()