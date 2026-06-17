import cv2
import os
import glob
from tqdm import tqdm

# ==========================================
# ⚙️ 설정 (경로만 형 환경에 맞게 수정해)
# ==========================================
INPUT_DIR = './AI/dataset/nturgb+d_rgb_C001_fall'  # 원본 영상 폴더
OUTPUT_DIR = './AI/dataset/expanded_videos'       # 확장된 영상 저장 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1프레임을 몇 번 복제할지 설정
FRONT_PADDING_COUNT = 60 

def expand_video(v_path):
    filename = os.path.basename(v_path)
    cap = cv2.VideoCapture(v_path)
    
    # 영상 정보 추출
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 모든 프레임 일단 리스트에 담기
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return

    # 💡 마술 로직: 1프레임 60개 복제 + 2프레임부터 끝까지
    # (결과적으로 10프레임 영상은 60 + 9 = 69프레임이 됨. 형 말대로 70개 맞추려면 61개 넣으면 돼)
    new_frames = [frames[0]] * FRONT_PADDING_COUNT + frames[1:]

    # 확장된 영상 저장
    out_path = os.path.join(OUTPUT_DIR, filename)
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in new_frames:
        out_writer.write(f)
    out_writer.release()

# ==========================================
# 🚀 실행
# ==========================================
video_files = glob.glob(os.path.join(INPUT_DIR, "*.avi"))
print(f"총 {len(video_files)}개의 영상을 변환하기 시작한다?")

for v_path in tqdm(video_files):
    expand_video(v_path)

print(f"\n✅ 마술 완료! 모든 영상이 '{OUTPUT_DIR}' 폴더에 저장됐어.")