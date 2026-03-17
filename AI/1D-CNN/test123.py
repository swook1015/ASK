import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 1. 경로 설정 (성욱 님 환경에 맞게 수정)
model_path = './AI/models/lpn_paper_3d_lifter.h5'
input_2d_path = './AI/dataset/LPN-train/ntu_lpn_input_2d.npy'
target_3d_path = './AI/dataset/LPN-train/ntu_lpn_target_3d_final.npy'

# 2. 모델 및 데이터 로드
model = tf.keras.models.load_model(model_path)
data_2d = np.load(input_2d_path)   # (N, 26)
data_3d_gt = np.load(target_3d_path) # (N, 39) 실제 정답(Ground Truth)

# 3. 뼈대 연결 (성욱 님의 TARGET_INDICES 순서에 100% 맞춤)
# 0:머리, 1:왼어깨, 2:오른어깨, 3:왼팔꿈치, 4:오른팔꿈치, 5:왼손목, 6:오른손목
# 7:왼골반, 8:오른골반, 9:왼무릎, 10:오른무릎, 11:왼발목, 12:오른발목
EDGES = [
    (0, 1), (0, 2), (1, 2),   # 상체(머리-어깨)
    (1, 3), (3, 5),           # 왼팔
    (2, 4), (4, 6),           # 오른팔
    (1, 7), (2, 8), (7, 8),   # 몸통(어깨-골반)
    (7, 9), (9, 11),          # 왼다리
    (8, 10), (10, 12)         # 오른다리
]

current_idx = 0

def draw_comparison(idx):
    plt.clf()
    # 예측 수행
    sample_2d = data_2d[idx:idx+1]
    pred_3d = model.predict(sample_2d, verbose=0)[0].reshape(13, 3)
    gt_3d = data_3d_gt[idx].reshape(13, 3)
    in_2d = data_2d[idx].reshape(13, 2)

    # --- 1. 왼쪽: LPN 전 (Input 2D) ---
    ax1 = fig.add_subplot(131)
    ax1.scatter(in_2d[:, 0], in_2d[:, 1], c='blue', s=20)
    for e in EDGES:
        ax1.plot([in_2d[e[0],0], in_2d[e[1],0]], [in_2d[e[0],1], in_2d[e[1],1]], c='gray', alpha=0.5)
    ax1.set_title(f"Input 2D (Frame {idx})")
    ax1.set_xlim([-1.1, 1.1]); ax1.set_ylim([1.1, -1.1]) # Y축 반전

    # --- 2. 가운데: LPN 후 (Predicted 3D) ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c='red', s=20)
    for e in EDGES:
        ax2.plot([pred_3d[e[0],0], pred_3d[e[1],0]], [pred_3d[e[0],1], pred_3d[e[1],1]], [pred_3d[e[0],2], pred_3d[e[1],2]], c='red')
    ax2.set_title("LPN Predicted 3D")
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    ax2.view_init(elev=20, azim=30)

    # --- 3. 오른쪽: 실제 정답 (Ground Truth 3D) ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c='green', s=20)
    for e in EDGES:
        ax3.plot([gt_3d[e[0],0], gt_3d[e[1],0]], [gt_3d[e[0],1], gt_3d[e[1],1]], [gt_3d[e[0],2], gt_3d[e[1],2]], c='green')
    ax3.set_title("Actual Ground Truth")
    ax3.set_xlim([-1, 1]); ax3.set_ylim([-1, 1]); ax3.set_zlim([-1, 1])
    ax3.view_init(elev=20, azim=30)

    plt.tight_layout()
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == ' ':
        current_idx = np.random.randint(0, len(data_2d)) # 랜덤하게 샘플링
        draw_comparison(current_idx)
    elif event.key == 'escape':
        plt.close()

fig = plt.figure(figsize=(18, 6))
fig.canvas.mpl_connect('key_press_event', on_key)

print("🎹 [Space]: 랜덤 샘플 비교 | [ESC]: 종료")
draw_comparison(current_idx)
plt.show()