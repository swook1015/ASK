import cv2
import numpy as np
import time
import subprocess
import psutil
from threading import Thread, Lock
from flask import Flask, Response
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

app = Flask(__name__)

# ==========================================
# ⚙️ 1. 당시 성공 세팅 (MoveNet + 360p)
# ==========================================
CAMERA_PORT = '/dev/video1' 
POSE_MODEL  = '/home/mendel/AI/mspte.tflite' # 당시 사용하던 MoveNet Thunder

W, H = 640, 360
FRAME_SIZE = W * H * 3
MIN_CONF = 0.3

# ==========================================
# 📹 2. 핵심: FFmpeg Subprocess 파이프 (30FPS의 비결)
# ==========================================
class CameraWorker:
    def __init__(self):
        # NC20의 1080p를 받아서 FFmpeg가 하드웨어적으로 360p로 줄여서 파이썬에 던짐
        self.cmd = [
            'ffmpeg', '-f', 'v4l2', '-input_format', 'mjpeg', 
            '-video_size', '1920x1080', '-framerate', '30', '-i', CAMERA_PORT,
            '-vf', f'scale={W}:{H}', '-f', 'image2pipe', 
            '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        self.pipe = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE*3)
        self.frame = None
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while True:
            # 파이프에서 640x360 이미지를 한 장씩 읽어옴
            raw_image = self.pipe.stdout.read(FRAME_SIZE)
            if len(raw_image) == FRAME_SIZE:
                image = np.frombuffer(raw_image, dtype=np.uint8).reshape((H, W, 3))
                with self.lock:
                    self.frame = image
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# ==========================================
# 🧠 3. MoveNet 추론 (결과값이 작아 CPU 부하 없음)
# ==========================================
class AIWorker:
    def __init__(self, model_path, camera):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.cam = camera
        self.p_size = common.input_size(self.interpreter)
        self.out_idx = self.interpreter.get_output_details()[0]['index']
        
        self.latest_res = {"frame": None, "kps": None, "fps": 0.0}
        self.res_lock = Lock()

    def start(self):
        Thread(target=self.infer, daemon=True).start()
        return self

    def infer(self):
        last_time = time.perf_counter()
        while True:
            frame = self.cam.get_frame()
            if frame is None:
                time.sleep(0.01); continue

            # 전처리 및 TPU 추론
            input_img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.p_size)
            common.set_input(self.interpreter, input_img)
            self.interpreter.invoke()
            
            # 결과 파싱 (딱 17개 관절이라 매우 가벼움)
            kps = self.interpreter.get_tensor(self.out_idx)[0][0]

            # FPS 계산
            curr_time = time.perf_counter()
            fps = 1.0 / (curr_time - last_time + 1e-6)
            last_time = curr_time

            with self.res_lock:
                self.latest_res = {"frame": frame, "kps": kps, "fps": fps}

# 객체 생성 및 실행
cam = CameraWorker().start()
ai = AIWorker(POSE_MODEL, cam).start()

# ==========================================
# 🖼️ 4. 웹 송출 로직
# ==========================================
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with ai.res_lock:
                res = ai.latest_res.copy()
            
            f = res["frame"]
            if f is None:
                time.sleep(0.01); continue
            
            # 관절 그리기
            kps = res["kps"]
            if kps is not None:
                for i in range(17):
                    y, x, conf = kps[i]
                    if conf > MIN_CONF:
                        cv2.circle(f, (int(x*W), int(y*H)), 5, (0, 0, 255), -1)

            cv2.putText(f, f"FPS: {res['fps']:.1f}", (10, 30), 1, 1.5, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', f)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<body style='margin:0;background:black;'><img src='/video_feed' style='width:100%;height:auto;'></body>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, threaded=True)