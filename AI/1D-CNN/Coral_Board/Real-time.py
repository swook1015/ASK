import cv2
import numpy as np
import time
import subprocess
import psutil
from threading import Thread
from queue import Queue
from flask import Flask, Response
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

app = Flask(__name__)

# ==========================================
# ⚙️ 설정
# ==========================================
CAMERA_PORT = '/dev/video1' 
POSE_MODEL = '/home/mendel/ask/msp.tflite'
# 🔥 중요: Raw 모델 말고, 예전에 정규화해서 학습했던 원래 모델을 쓰세요!
CNN_MODEL = '/home/mendel/ask/edgetpu.tflite' 

W, H = 640, 360
FRAME_SIZE = W * H * 3
ALPHA = 0.15          
MIN_CONF = 0.25       
DRAW_CONF = 0.3       
FALL_THRESHOLD = 0.50 
WINDOW_SIZE = 20

EDGES = [(0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (7,9), (6,8), (8,10), 
         (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

class CameraWorker:
    def __init__(self):
        cmd = [
            'ffmpeg', '-loglevel', 'quiet', '-f', 'v4l2', 
            '-input_format', 'mjpeg', '-video_size', '1280x720', 
            '-framerate', '30', '-i', CAMERA_PORT, 
            '-vf', f'scale={W}:{H}', '-f', 'image2pipe', 
            '-probesize', '32', '-analyzeduration', '0', 
            '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        self.pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE * 3)
        self.q = Queue(maxsize=3)
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            raw = self.pipe.stdout.read(FRAME_SIZE)
            if len(raw) == FRAME_SIZE:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
                if not self.q.full(): self.q.put(frame)

    def read(self):
        return self.q.get()

p_int = make_interpreter(POSE_MODEL); p_int.allocate_tensors()
c_int = make_interpreter(CNN_MODEL); c_int.allocate_tensors()

p_size = common.input_size(p_int)
p_out_idx = p_int.get_output_details()[0]['index']
c_in_details = c_int.get_input_details()[0]
c_out_details = c_int.get_output_details()[0]
in_scale, in_zp = c_in_details['quantization']
out_scale, out_zp = c_out_details['quantization']

cam = CameraWorker().start()
output_frame = None

def detect_fall():
    global output_frame
    buffer = []
    score_history = [] 
    prev_smoothed = None 
    fps_avg = 0
    tpu_ms = 0 
    
    print(f"🎬 실시간 카메라 분석 시작: {CAMERA_PORT}")

    while True:
        start_time = time.perf_counter()
        frame = cam.read()
        display_frame = frame.copy()
        
        # 1. MoveNet 추론
        input_frame = cv2.resize(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), p_size)
        common.set_input(p_int, input_frame)
        p_int.invoke()
        kps = p_int.get_tensor(p_out_idx)[0][0] 
        ys, xs, cs = kps[:, 0], kps[:, 1], kps[:, 2]

        # 2. 🔥 [1단계 보정] 떨림 제거 (EMA)
        current_51_raw = []
        if prev_smoothed is None:
            prev_smoothed = np.zeros(51)
            for j in range(17):
                prev_smoothed[j*3] = xs[j]
                prev_smoothed[j*3+1] = ys[j]

        for j in range(17):
            if cs[j] > MIN_CONF:
                s_x = (xs[j] * ALPHA) + (prev_smoothed[j*3] * (1 - ALPHA))
                s_y = (ys[j] * ALPHA) + (prev_smoothed[j*3+1] * (1 - ALPHA))
            else:
                s_x, s_y = prev_smoothed[j*3], prev_smoothed[j*3+1]
                
            current_51_raw.extend([s_x, s_y, cs[j]])

        current_51_raw = np.array(current_51_raw)
        prev_smoothed = current_51_raw

        # 3. 🎨 시각화 (뼈대 그리기 - CPU 먹는 바운딩 박스 제거!)
        for edge in EDGES:
            p1, p2 = edge
            x1, y1, c1 = current_51_raw[p1*3], current_51_raw[p1*3+1], current_51_raw[p1*3+2]
            x2, y2, c2 = current_51_raw[p2*3], current_51_raw[p2*3+1], current_51_raw[p2*3+2]
            
            if c1 > DRAW_CONF and c2 > DRAW_CONF:
                cv2.line(display_frame, (int(x1*W), int(y1*H)), (int(x2*W), int(y2*H)), (0, 255, 0), 2)

        # 4. 🔥 [2단계 보정] 모델에 넣기 전 정규화 (학습 데이터와 완벽 일치)
        # 떨림이 잡힌 좌표를 기준으로 중심(cx, cy)과 크기(mr)를 구합니다.
        valid_mask = current_51_raw[2::3] > 0.1
        current_51_norm = np.zeros(51)
        
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

        # 5. CNN 낙상 판단 및 TPU 속도 측정
        # 🔥 중요: 모델에는 반드시 정규화된(norm) 데이터를 넣습니다!
        buffer.append(current_51_norm)
        if len(buffer) > WINDOW_SIZE: buffer.pop(0)
        
        score = 0
        if len(buffer) == WINDOW_SIZE:
            raw_seq = np.array(buffer, dtype=np.float32)
            input_quant = (raw_seq[np.newaxis, ..., np.newaxis] / in_scale + in_zp).astype(np.int8)
            common.set_input(c_int, input_quant)
            
            tpu_start = time.perf_counter()
            c_int.invoke()
            tpu_ms = (time.perf_counter() - tpu_start) * 1000
            
            raw_score = common.output_tensor(c_int, 0)[0][0]
            score = (raw_score.astype(np.float32) - out_zp) * out_scale

        score_history.append(score)
        if len(score_history) > 5: score_history.pop(0)
        avg_score = sum(score_history) / len(score_history)

        # 6. UI 오버레이
        end_time = time.perf_counter()
        fps_avg = fps_avg * 0.9 + (1.0 / (end_time - start_time)) * 0.1
        cpu_usage = psutil.cpu_percent() 

        cv2.rectangle(display_frame, (0, 0), (W, 35), (30, 30, 30), -1)
        text_color = (0, 0, 255) if avg_score > FALL_THRESHOLD else (255, 255, 255)
        
        info_text = f"FPS: {fps_avg:.1f} | CPU: {cpu_usage}% | TPU: {tpu_ms:.1f}ms | Score: {avg_score:.2f}"
        cv2.putText(display_frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1 if avg_score <= FALL_THRESHOLD else 2)

        if avg_score > FALL_THRESHOLD:
            cv2.rectangle(display_frame, (0, 0), (W, H), (0, 0, 255), 8)
            cv2.putText(display_frame, "!!! FALL DETECTED !!!", (W//2 - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        output_frame = display_frame

def generate():
    global output_frame
    while True:
        if output_frame is None: continue
        _, encoded = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')

@app.route("/video_feed")
def video_feed(): return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/")
def index(): return "<h1>Coral TPU Real-time Fall Detection</h1><img src='/video_feed'>"

if __name__ == "__main__":
    Thread(target=detect_fall, daemon=True).start()
    app.run(host="0.0.0.0", port=5050, threaded=True)