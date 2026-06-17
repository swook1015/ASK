import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
npy_path = './AI/dataset/urfd_processed/urfd_3d_train_X.npy' 
try:
    data = np.load(npy_path).astype(np.float32)
    print(f"📦 데이터 로드 완료: {data.shape}")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {npy_path}")
    exit()

# [핵심] 3D 좌표 형태로 복원 (2,880프레임, 13개 관절, x/y/z)
data_3d = data.reshape(-1, 13, 3) 
num_total_frames = data_3d.shape[0]
print(f"✅ 총 프레임 수: {num_total_frames}")

# ==========================================
# 2. 뼈대 연결 정보 (MediaPipe 13개 관절 기준)
# ==========================================
# 0:코, 1:왼어깨, 2:오어깨, 3:왼팔꿈치, 4:오팔꿈치, 5:왼손목, 6:오손목, 
# 7:왼골반, 8:오골반, 9:왼무릎, 10:오무릎, 11:왼발목, 12:오발목
bones = [
    (1, 2), (1, 3), (3, 5), # 어깨, 왼팔
    (2, 4), (4, 6),         # 오른팔
    (1, 7), (2, 8),         # 몸통
    (7, 8), (7, 9), (9, 11),# 골반, 왼다리
    (8, 10), (10, 12)       # 오른다리
]

# ==========================================
# 3. 3D 애니메이션 설정 및 실행
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.cla() # 이전 화면 지우기
    
    # 현재 프레임 좌표 추출
    x = data_3d[frame, :, 0]
    y = data_3d[frame, :, 1]
    z = data_3d[frame, :, 2]
    
    # 3D 관절(점) 시각화
    ax.scatter(x, z, -y, c='red', marker='o', s=50) # y축 반전(-y)으로 상하 정렬
    
    # 뼈대(선) 시각화
    for bone in bones:
        p1, p2 = bone
        ax.plot([x[p1], x[p2]], [z[p1], z[p2]], [-y[p1], -y[p2]], 
                color='blue', linewidth=2, alpha=0.7)
    
    # 그래프 범위 설정 (좌표값이 작을 경우 -1~1, 클 경우 -2~2로 조정)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    ax.set_xlabel('X (Left/Right)')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Up/Down)')
    ax.set_title(f"URFD 3D Validation - Frame: {frame}/{num_total_frames}")

# 애니메이션 실행 (frames 인자에 실제 프레임 수 입력)
ani = animation.FuncAnimation(fig, update, frames=num_total_frames, 
                              interval=30, repeat=True)

plt.tight_layout()
plt.show()