import cv2
import numpy as np
import time
import subprocess
from threading import Thread
from queue import Queue
import tflite_runtime.interpreter as tflite
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# 설정
CAMERA_PORT = '/dev/video0'
POSE_MODEL = '/home/mendel/ask/msp.tflite'
CNN_MODEL = 'fall_model_conv1d_edgetpu.tflite' # 🌟 새로 만든 파일명
W, H = 640, 360
FRAME_SIZE = W * H * 3

# [스레드 1] FFmpeg MJPEG 캡처
class CameraWorker:
    def __init__(self):
        cmd = ['ffmpeg', '-loglevel', 'quiet', '-f', 'v4l2', '-input_format', 'mjpeg', 
               '-video_size', '1280x720', '-framerate', '30', '-i', CAMERA_PORT, 
               '-vf', f'scale={W}:{H}', '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
        self.pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE * 3)
        self.q = Queue(maxsize=3)
        self.stopped = False
    def start(self): Thread(target=self.update, daemon=True).start(); return self
    def update(self):
        while not self.stopped:
            raw = self.pipe.stdout.read(FRAME_SIZE)
            if len(raw) == FRAME_SIZE:
                if not self.q.full(): self.q.put(np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3)))
    def read(self): return self.q.get()

# 엔진 초기화
p_int = make_interpreter(POSE_MODEL); p_int.allocate_tensors()
c_int = make_interpreter(CNN_MODEL); c_int.allocate_tensors()
p_size = common.input_size(p_int); p_out = p_int.get_output_details()
c_in, c_out = c_int.get_input_details()[0]['index'], c_int.get_output_details()[0]['index']

cam = CameraWorker().start()
buffer = []
prev = np.zeros(51, dtype=np.float32)
f_cnt = 0; start = time.time()

print("🚀 30fps 낙상 감지 가동!")
try:
    while True:
        frame = cam.read()
        f_cnt += 1
        
        # 1. PoseNet (TPU)
        common.set_input(p_int, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), p_size))
        p_int.invoke()
        kps = p_int.get_tensor(p_out[0]['index'])[0][0]
        
        # 2. 좌표 전처리 (EMA 스무딩 + 정규화)
        coords = []
        xs, ys, cs = kps[:, 1], kps[:, 0], kps[:, 2] # x, y, conf
        valid = cs > 0.15
        if np.any(valid):
            cx, cy = np.mean(xs[valid]), np.mean(ys[valid])
            mr = max(np.max(xs[valid])-np.min(xs[valid]), np.max(ys[valid])-np.min(ys[valid])) + 1e-6
            for j in range(17):
                nx = (xs[j]-cx)/mr if cs[j]>0.15 else 0
                ny = (ys[j]-cy)/mr if cs[j]>0.15 else 0
                coords.extend([nx, ny, cs[j]])
        else: coords = [0]*51

        # 3. 1D-CNN 추론 (TPU)
        buffer.append(coords)
        if len(buffer) >= 20:
            c_int.set_tensor(c_in, np.expand_dims(np.array(buffer[-20:], dtype=np.float32), 0))
            c_int.invoke()
            score = c_int.get_tensor(c_out)[0][0]
            if score > 0.7: print(f"🚨 낙상!! ({score:.2f})")
            if len(buffer) > 40: buffer.pop(0)

        if f_cnt % 30 == 0:
            print(f"⚡ 속도: {f_cnt/(time.time()-start):.1f} FPS")

except KeyboardInterrupt: pass