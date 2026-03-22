import numpy as np

data = np.load('./AI/dataset/LPN-train/ntu_lpn_target_3d_final.npy') # (3412800, 39)

# 영상 100개 분량 추출 (100 * 60 = 6000)
num_samples = 100
extract_frames = num_samples * 60
extracted_data = data[:extract_frames]

# 영점 조절 시 모델 입력 규격에 맞게 다시 묶기
final_data = extracted_data.reshape(num_samples, 60, 39)

np.save('./calibration_100_samples.npy', final_data)
print(f"✅ 추출 완료! 모양: {final_data.shape}") # (100, 60, 39)
