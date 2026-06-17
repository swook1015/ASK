import cv2
import os
import glob
from tqdm import tqdm

# ==========================================
# ⚙️ 1. 경로 설정
# ==========================================
INPUT_DIR = './AI/dataset/nturgb+d_rgb_C001'
OUTPUT_DIR = './AI/dataset/nturgb+d_rgb_C001_padded' # 변환된 영상이 저장될 새 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 60

# 영상 파일 목록 불러오기 (avi 확장자)
video_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.avi")))

print(f"🚀 총 {len(video_files)}개의 영상에 대해 프레임 패딩 작업을 시작합니다...")

for v_path in tqdm(video_files, desc="영상 변환 중"):
    filename = os.path.basename(v_path)
    out_path = os.path.join(OUTPUT_DIR, filename)

    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 1. 원본 영상의 전체 프레임 읽어오기
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if not frames:
        continue

    # ==========================================
    # 🧠 2. 핵심 로직: 프레임 패딩 (논문 방식 적용)
    # ==========================================
    # [설명] frames[0]을 59번 복사(WINDOW_SIZE-1)하여 앞에 붙임.
    # 이렇게 하면 기존 1프레임이 총 60프레임 동안 정지 화면처럼 유지되고,
    # 기존 2프레임(frames[1])은 정확히 61번째 프레임이 됩니다.
    
    padded_frames = [frames[0]] * (WINDOW_SIZE - 1) + frames
    
    # 💡 (선택) 1D-TCN의 끝부분 평가를 위해 영상 마지막 프레임도 60프레임 늘려줍니다.
    # 형이 짜두었던 실시간 추론 파이프라인과 완벽히 동일한 조건입니다.
    padded_frames += [frames[-1]] * WINDOW_SIZE

    # ==========================================
    # 💾 3. 새로운 영상 파일로 저장
    # ==========================================
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for f in padded_frames:
        out_writer.write(f)
    out_writer.release()

print("\n" + "="*50)
print(f"🎉 변환 완료! 패딩된 영상들이 아래 경로에 저장되었습니다.")
print(f"📂 경로: {OUTPUT_DIR}")
print("="*50)