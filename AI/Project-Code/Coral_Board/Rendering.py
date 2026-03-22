import cv2
import numpy as np
import time
import os
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# ==========================================
# ⚙️ 설정
# ==========================================
INPUT_VIDEO_PATH = 'test_video.mp4'    
OUTPUT_VIDEO_PATH = 'result_video.avi' 

POSE_MODEL = '/home/mendel/ask/msp.tflite'
# 🔥 중요 1: Raw 모델 말고, "원래 모델(정규화 학습)"을 쓰세요!
CNN_MODEL = '/home/mendel/ask/edgetpu.tflite' 

W, H = 640, 360  
ALPHA = 0.15     
MIN_CONF = 0.05       # 🔥 중요 2: 0.25 -> 0.05로 하향! (쓰러질 때 얼어붙지 않게)
FALL_THRESHOLD = 0.50 
WINDOW_SIZE = 20

EDGES = [(0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (7,9), (6,8), (8,10), 
         (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

p_int = make_interpreter(POSE_MODEL); p_int.allocate_tensors()
c_int = make_interpreter(CNN_MODEL); c_int.allocate_tensors()

p_size = common.input_size(p_int)
p_out_idx = p_int.get_output_details()[0]['index']
c_in_details = c_int.get_input_details()[0]
c_out_details = c_int.get_output_details()[0]
in_scale, in_zp = c_in_details['quantization']
out_scale, out_zp = c_out_details['quantization']

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W, H))

def process_video():
    buffer = []
    prev_smoothed = None 
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 영상 분석 시작: {INPUT_VIDEO_PATH}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        display_frame = cv2.resize(frame, (W, H))
        
        # 1. MoveNet 추론
        input_frame = cv2.resize(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), p_size)
        common.set_input(p_int, input_frame)
        p_int.invoke()
        kps = p_int.get_tensor(p_out_idx)[0][0] 
        ys, xs, cs = kps[:, 0], kps[:, 1], kps[:, 2]

        # 2. [1단계 보정] 떨림 제거(EMA) 및 X,Y 순서 교정
        current_51_raw = []
        if prev_smoothed is None:
            prev_smoothed = np.zeros(51)
            for j in range(17):
                prev_smoothed[j*3] = xs[j]
                prev_smoothed[j*3+1] = ys[j]

        for j in range(17):
            if cs[j] > MIN_CONF: # 0.05 이상이면 어떻게든 따라감!
                s_x = (xs[j] * ALPHA) + (prev_smoothed[j*3] * (1 - ALPHA))
                s_y = (ys[j] * ALPHA) + (prev_smoothed[j*3+1] * (1 - ALPHA))
            else:
                s_x, s_y = prev_smoothed[j*3], prev_smoothed[j*3+1]
                
            current_51_raw.extend([s_x, s_y, cs[j]])

        current_51_raw = np.array(current_51_raw)
        prev_smoothed = current_51_raw

        # 3. 뼈대 그리기 (시각화는 깔끔하게 0.3 이상만 그림)
        for edge in EDGES:
            p1, p2 = edge
            x1, y1, c1 = current_51_raw[p1*3], current_51_raw[p1*3+1], current_51_raw[p1*3+2]
            x2, y2, c2 = current_51_raw[p2*3], current_51_raw[p2*3+1], current_51_raw[p2*3+2]
            
            if c1 > 0.3 and c2 > 0.3: # 👈 그리는 건 엄격하게! (뭉침 방지)
                cv2.line(display_frame, (int(x1*W), int(y1*H)), (int(x2*W), int(y2*H)), (0, 255, 0), 2)

        # 4. 🔥 [2단계 보정] 모델 입력용 "정규화" (서 있을 때 낙상 뜨는 것 방지)
        current_51_norm = np.zeros(51)
        valid_mask = current_51_raw[2::3] > 0.1 # 신뢰도 0.1 이상인 관절만 사용해 중심점 계산
        
        if np.any(valid_mask):
            valid_x = current_51_raw[0::3][valid_mask]
            valid_y = current_51_raw[1::3][valid_mask]
            cx, cy = np.mean(valid_x), np.mean(valid_y)
            mr = max(np.max(valid_x) - np.min(valid_x), np.max(valid_y) - np.min(valid_y)) + 1e-6
            
            for j in range(17):
                current_51_norm[j*3] = (current_51_raw[j*3] - cx) / mr
                current_51_norm[j*3+1] = (current_51_raw[j*3+1] - cy) / mr
                current_51_norm[j*3+2] = current_51_raw[j*3+2]
        else:
            current_51_norm = current_51_raw.copy()

        # 5. CNN 낙상 판단 (정규화된 데이터를 밀어넣음)
        buffer.append(current_51_norm)
        if len(buffer) > WINDOW_SIZE: buffer.pop(0)
        
        score = 0
        if len(buffer) == WINDOW_SIZE:
            raw_seq = np.array(buffer, dtype=np.float32)
            input_quant = (raw_seq[np.newaxis, ..., np.newaxis] / in_scale + in_zp).astype(np.int8)
            common.set_input(c_int, input_quant)
            c_int.invoke()
            raw_score = common.output_tensor(c_int, 0)[0][0]
            score = (raw_score.astype(np.float32) - out_zp) * out_scale

        # 6. 정보 오버레이
        cv2.rectangle(display_frame, (0, 0), (W, 35), (30, 30, 30), -1)
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} | Score: {score:.2f}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if score > FALL_THRESHOLD:
            cv2.rectangle(display_frame, (0, 35), (W, H), (0, 0, 255), 4)
            cv2.putText(display_frame, "FALL DETECTED", (W-200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(display_frame)

        if frame_count % 50 == 0:
            print(f"🔄 진행 중... ({frame_count}/{total_frames})")

    cap.release()
    out.release()
    print(f"✅ 분석 완료! 저장된 파일: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()