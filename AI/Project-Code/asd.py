import numpy as np

# 데이터 로드
file_path = './AI/dataset/LPN-train/ntu_lpn_input_2d_final.npy'
print(f"📦 데이터 로드 중...")
X = np.load(file_path)

# 통계적 유의미성을 위해 10만 프레임(약 55분 분량의 동작) 샘플링
sample_size = min(100000, len(X))
samples = X[:sample_size].reshape(sample_size, 13, 2)

x_coords = samples[:, :, 0]
y_coords = samples[:, :, 1]

print("\n" + "="*50)
print(f"🔍 [대규모 데이터({sample_size} 프레임) 정규화 공식 검증]")
print("="*50)

# 1. 전체 데이터 범위 확인
print(f"▶ 1. 전체 X 좌표 범위: {np.min(x_coords):.4f} ~ {np.max(x_coords):.4f}")
print(f"▶ 1. 전체 Y 좌표 범위: {np.min(y_coords):.4f} ~ {np.max(y_coords):.4f}")

# 2. 골반(Root) 중심 검증 (Index 7: 좌측, Index 8: 우측)
left_hips = samples[:, 7, :]
right_hips = samples[:, 8, :]
root_centers = (left_hips + right_hips) / 2.0
avg_root_x = np.mean(root_centers[:, 0])
avg_root_y = np.mean(root_centers[:, 1])

print(f"\n▶ 2. 평균 골반 중심 좌표 (X, Y): ({avg_root_x:.6f}, {avg_root_y:.6f})")
if abs(avg_root_x) < 0.05 and abs(avg_root_y) < 0.05:
    print("  💡 [분석] 완벽한 (0, 0)입니다. 골반을 중심축으로 한 상대 좌표(Root-relative)가 확실합니다.")
else:
    print("  ⚠️ [경고] (0, 0)이 아닙니다. 화면 비율 등 다른 정규화를 사용했을 수 있습니다.")

# 3. Y축 반전(물구나무) 검증 (Index 0: 코, Index 11/12: 발목)
noses_y = samples[:, 0, 1]
ankles_y = (samples[:, 11, 1] + samples[:, 12, 1]) / 2.0
avg_nose_y = np.mean(noses_y)
avg_ankle_y = np.mean(ankles_y)

print(f"\n▶ 3. 평균 코(머리) Y좌표: {avg_nose_y:.4f}")
print(f"▶ 3. 평균 발목 Y좌표   : {avg_ankle_y:.4f}")
if avg_nose_y > avg_ankle_y:
    print("  💡 [분석] 머리가 양수(+)이고 발목이 음수(-)입니다. 카메라 픽셀과 Y축 위아래가 100% 반대입니다.")
else:
    print("  ⚠️ [경고] 카메라 픽셀과 방향이 같습니다. Y축을 뒤집으면 안 됩니다.")

# 4. 스케일(사람 키) 검증
heights = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)
avg_height = np.mean(heights)

print(f"\n▶ 4. 평균 사람 키 (Y축 최대-최소 길이): {avg_height:.4f}")
print("  💡 [분석] 이 수치가 우리가 웹캠 픽셀을 나눌 때 사용할 '진짜 분모(스케일)' 값입니다.")
print("="*50 + "\n")