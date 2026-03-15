import numpy as np

# 원본 로드
data = np.load('./AI/dataset/LPN-label/ntu_lpn_target_3d.npy')

# 딱 100개만 슬라이싱해서 저장
sample_data = data[:100]
np.save('lpn_test_sample.npy', sample_data)

print("✅ lpn_test_sample.npy 생성 완료! 이 파일을 코랩에 올리세요.")