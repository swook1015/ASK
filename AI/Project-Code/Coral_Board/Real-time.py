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
# ⚙️ 1. 설정
# ==========================================
CAMERA_PORT = '/dev/video1' 
POSE_MODEL  = '/home/mendel/AI/yolov8n-pose_full_integer_quant_edgetpu.tflite'

W, H = 640, 360
FRAME_SIZE = W * H * 3
MIN_CONF = 0.20
ALPHA = 0.8  

EDGES = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
         (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

# ==========================================
# 📹 2. 카메라 스레드 (ffmpeg 파이프라인 복구)
# ==========================================
class CameraWorker:
    def __init__(self):
        cmd = [
            'ffmpeg', '-loglevel', 'quiet', '-f', 'v4l2', 
            '-input_format', 'mjpeg', '-video_size', '1280x720', 
            '-framerate', '30', '-i', CAMERA_PORT, 
            '-vf', f'scale={W}:{H}', '-f', 'image2pipe', 
            '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        self.pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE * 3)
        self.frame = None
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while True:
            raw = self.pipe.stdout.read(FRAME_SIZE)
            if len(raw) == FRAME_SIZE:
                tmp = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
                with self.lock:
                    self.frame = tmp
            else:
                time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# ==========================================
# 🧠 3. AI 추론 스레드 (TPU 전담)
# ==========================================
class InferenceThread:
    def __init__(self, model_path, camera):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.cam = camera
        self.p_size = common.input_size(self.interpreter)
        self.p_out_idx = self.interpreter.get_output_details()[0]['index']
        self.p_scale, self.p_zp = self.interpreter.get_output_details()[0]['quantization']
        
        self.latest_res = {"frame": None, "kps": None, "tpu_pct": 0.0, "ai_fps": 0.0}
        self.res_lock = Lock()
        self.prev_kps = None

    def start(self):
        Thread(target=self.infer, daemon=True).start()
        return self

    def infer(self):
        avg_fps = 0
        while True:
            loop_start = time.perf_counter()
            frame = self.cam.get_frame()
            if frame is None:
                time.sleep(0.005); continue

            # YOLO 전처리 및 추론
            inp = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.p_size)
            common.set_input(self.interpreter, inp)
            
            t1 = time.perf_counter()
            self.interpreter.invoke()
            t_tpu = time.perf_counter() - t1
            
            # 결과 파싱
            raw_out = self.interpreter.get_tensor(self.p_out_idx)
            if raw_out.shape[1] > raw_out.shape[2]: 
                raw_out = np.transpose(raw_out, (0, 2, 1))
            
            out_f = (raw_out.astype(np.float32) - self.p_zp) * self.p_scale
            preds = out_f[0]
            best_idx = np.argmax(preds[4, :])
            conf = preds[4, best_idx]

            raw_kps = None
            if conf > MIN_CONF:
                raw_kps = np.zeros((17, 3))
                best_kps = preds[5:, best_idx]
                is_px = np.max(best_kps[0::3]) > 2.0
                
                for i in range(17):
                    kx, ky, kc = best_kps[i*3], best_kps[i*3+1], best_kps[i*3+2]
                    raw_kps[i, 0] = (ky / self.p_size[1]) if is_px else ky
                    raw_kps[i, 1] = (kx / self.p_size[0]) if is_px else kx
                    raw_kps[i, 2] = kc

                if self.prev_kps is not None:
                    raw_kps[:, :2] = raw_kps[:, :2] * ALPHA + self.prev_kps[:, :2] * (1-ALPHA)
                self.prev_kps = raw_kps.copy()
            else:
                self.prev_kps = None

            loop_time = time.perf_counter() - loop_start + 1e-6
            tpu_pct = (t_tpu / loop_time) * 100
            
            current_fps = 1.0 / loop_time
            avg_fps = avg_fps * 0.9 + current_fps * 0.1

            with self.res_lock:
                self.latest_res = {"frame": frame, "kps": raw_kps, "tpu_pct": tpu_pct, "ai_fps": avg_fps}

    def get_latest(self):
        with self.res_lock:
            return self.latest_res.copy()

# 객체 초기화
cam_worker = CameraWorker().start()
ai_engine = InferenceThread(POSE_MODEL, cam_worker).start()

# ==========================================
# 🖼️ 4. 웹 스트리밍
# ==========================================
@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            r = ai_engine.get_latest()
            if r["frame"] is None:
                time.sleep(0.01); continue
                
            f = r["frame"]
            if r["kps"] is not None:
                kps = r["kps"]
                for e in EDGES:
                    p1, p2 = e
                    if kps[p1, 2] > MIN_CONF and kps[p2, 2] > MIN_CONF:
                        cv2.line(f, (int(kps[p1,1]*W), int(kps[p1,0]*H)), (int(kps[p2,1]*W), int(kps[p2,0]*H)), (0,255,0), 2)
                for p in kps:
                    if p[2] > MIN_CONF: cv2.circle(f, (int(p[1]*W), int(p[0]*H)), 4, (0,0,255), -1)

            cv2.rectangle(f, (0,0), (W, 45), (0,0,0), -1)
            cv2.putText(f, f"AI FPS: {r['ai_fps']:.1f} | CPU: {psutil.cpu_percent()}%", (10, 18), 1, 1, (255,255,255), 1)
            cv2.putText(f, f"TPU Util: {r['tpu_pct']:.1f}% | Mode: ffmpeg Multi-Thread", (10, 38), 1, 1, (0,255,255), 1)

            _, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + enc.tobytes() + b'\r\n')
            time.sleep(0.01)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index(): return "<body style='margin:0;background:black;'><img src='/video_feed' style='width:100%;height:auto;'></body>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True)