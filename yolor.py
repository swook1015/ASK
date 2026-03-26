import os
import cv2
import numpy as np
import time
import sys
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# ==========================================
# ⚙️ 1. 설정 및 유틸리티 함수
# ==========================================
POSE_MODEL  = '416yolov8n-pose_full_integer_quant_edgetpu.tflite'
LPN_MODEL   = 'lpn_seq_60_quant_edgetpu.tflite'
TCN_MODEL   = 'tcn_manual_fixed_edgetpu.tflite'

LPN_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NEW_EDGES = [(0,1), (0,2), (1,2), (1,3), (3,5), (2,4), (4,6), (1,7), (2,8), (7,8), (7,9), (9,11), (8,10), (10,12)]

def letterbox(im, new_shape=(416, 416), color=(114, 114, 114)):  
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)

def pure_numpy_nms(boxes, scores, threshold=0.45):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= threshold)[0] + 1]
    return keep

def fast_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

# ==========================================
# ⌨️ 2. 사용자 입력 처리 (cin 역할)
# ==========================================
print("=" * 60)
print("🎥 낙상 감지 비디오 분석 프로그램 🎥")
print("=" * 60)

while True:
    try:
        user_input = input("▶ 영상 파일 이름(확장자 제외)을 입력하세요 (종료: q) : ").strip()
        
        if user_input.lower() == 'q':
            print("프로그램을 종료합니다.")
            sys.exit(0)
            
        INPUT_VIDEO = f"{user_input}.mp4"
        
        if not os.path.exists(INPUT_VIDEO):
            print(f"🚨 에러: 현재 폴더에 '{INPUT_VIDEO}' 파일이 없습니다. 다시 확인해주세요.\n")
            continue
        
        break # 파일을 찾았으면 루프 탈출
        
    except KeyboardInterrupt:
        print("\n프로그램을 강제 종료합니다.")
        sys.exit(0)

OUTPUT_VIDEO = f'{INPUT_VIDEO}_result.avi'
LOG_FILE = f'{INPUT_VIDEO}_log.txt' 

# ==========================================
# 🧠 3. AI 모델 로드
# ==========================================
print("\n🚀 AI 모델 로드 중...")
pose_int = make_interpreter(POSE_MODEL); pose_int.allocate_tensors()
lpn_int  = make_interpreter(LPN_MODEL);  lpn_int.allocate_tensors()
tcn_int  = make_interpreter(TCN_MODEL);  tcn_int.allocate_tensors()

p_in_details = pose_int.get_input_details()[0]
in_scale, in_zp = p_in_details['quantization']
p_out_details = pose_int.get_output_details()[0]
p_out_idx = p_out_details['index']
p_scale, p_zp = p_out_details['quantization']

l_in_scale, l_in_zp = lpn_int.get_input_details()[0]['quantization']
l_out_idx, l_out_scale, l_out_zp = lpn_int.get_output_details()[0]['index'], *lpn_int.get_output_details()[0]['quantization']
t_in_idx, t_in_scale, t_in_zp = tcn_int.get_input_details()[0]['index'], *tcn_int.get_input_details()[0]['quantization']
t_out_idx, t_out_scale, t_out_zp = tcn_int.get_output_details()[0]['index'], *tcn_int.get_output_details()[0]['quantization']

cap = cv2.VideoCapture(INPUT_VIDEO)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS == 0 or np.isnan(FPS): FPS = 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

txt_log = open(LOG_FILE, 'w', encoding='utf-8')
txt_log.write(f"--- Fall Detection Log for {INPUT_VIDEO} ---\n")

joint_buffer = []
prev_kps_pixel = None
prev_scale = None 
smooth_score = 0.0 
fall_score = 0.0
status = "NORMAL"
ghost_frames = 0
MAX_GHOST = 45
draw_kps, confs_out = [], []
ALPHA = 0.1
avg_fps = 0.0
frame_idx = 0
last_time = time.perf_counter()

# ==========================================
# ⚙️ 4. 영상 처리 루프
# ==========================================
print(f"🎬 영상 처리 시작: {W}x{H} @ {FPS}FPS (총 {total_frames} 프레임)")
print("-" * 80)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_idx += 1
    st = time.perf_counter()
    
    img_p, ratio, (pw, ph) = letterbox(frame, (416, 416))
    img_rgb = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
    
    img_input = np.clip(np.round((img_rgb / 255.0) / in_scale + in_zp), -128, 127).astype(np.int8)
    common.set_input(pose_int, img_input)
    pose_int.invoke()
    
    raw_out = pose_int.get_tensor(p_out_idx)
    preds_raw = raw_out[0].transpose(1, 0) if raw_out.shape[1] == 56 else raw_out[0]
    preds_float = (preds_raw.astype(np.float32) - p_zp) * p_scale
    
    scores = fast_sigmoid(preds_float[:, 4])
    valid_mask = scores > 0.5 
    is_valid_person = False

    if np.any(valid_mask):
        v_preds, v_scores = preds_float[valid_mask], scores[valid_mask]
        v_preds[:, :4] *= 416.0
        boxes = np.column_stack((v_preds[:,0]-v_preds[:,2]/2, v_preds[:,1]-v_preds[:,3]/2, v_preds[:,2], v_preds[:,3]))
        nms_idx = pure_numpy_nms(boxes, v_scores, 0.45)
        
        if len(nms_idx) > 0:
            best_row = v_preds[nms_idx[0]]
            kps = best_row[5:].reshape(17, 3)
            kps[:, :2] *= 416.0
            kps[:, 2] = fast_sigmoid(kps[:, 2])
            
            target_kps = kps[LPN_INDICES]
            confs_out = target_kps[:, 2].tolist()
            
            avg_kp_conf = sum(confs_out) / len(confs_out)
            if avg_kp_conf >= 0.2:
                is_valid_person = True
                ghost_frames = 0
                xs, ys = (target_kps[:, 0] - pw) / ratio, (target_kps[:, 1] - ph) / ratio
                curr_pixel = np.stack((xs, ys), axis=-1)
                
                SMOOTH_KPS = 0.80
                if prev_kps_pixel is None: prev_kps_pixel = curr_pixel
                else: prev_kps_pixel = prev_kps_pixel * (1.0 - SMOOTH_KPS) + curr_pixel * SMOOTH_KPS
                    
                draw_kps = prev_kps_pixel.tolist()

                root_x = (prev_kps_pixel[7][0] + prev_kps_pixel[8][0]) / 2.0
                root_y = (prev_kps_pixel[7][1] + prev_kps_pixel[8][1]) / 2.0

                min_x = np.min(prev_kps_pixel[:, 0])
                max_x = np.max(prev_kps_pixel[:, 0])
                min_y = np.min(prev_kps_pixel[:, 1])
                max_y = np.max(prev_kps_pixel[:, 1])
                
                span_x = max_x - min_x
                span_y = max_y - min_y
                
                max_span = max(span_x, span_y) + 1e-6
                raw_scale = 1.6642 / max_span
                
                if prev_scale is None: prev_scale = raw_scale
                else: prev_scale = prev_scale * 0.5 + raw_scale * 0.5
                
                scale = prev_scale

                lpn_in = []
                for px, py in prev_kps_pixel:
                    lpn_in.extend([(px - root_x) * scale, -(py - root_y) * scale])
                
                joint_buffer.append(lpn_in)
                
                if len(joint_buffer) == 1:
                    joint_buffer = [joint_buffer[0]] * 59 + joint_buffer
                    
                if len(joint_buffer) > 60: joint_buffer.pop(0)

                if len(joint_buffer) == 60:
                    seq = np.array(joint_buffer, dtype=np.float32).reshape(1, 60, 26)
                    s_q = np.clip(np.round((seq / l_in_scale) + l_in_zp), -128, 127).astype(np.int8)
                    common.set_input(lpn_int, s_q); lpn_int.invoke()
                    
                    l_out_float = (lpn_int.get_tensor(l_out_idx).astype(np.float32) - l_out_zp) * l_out_scale
                    t_in_q = np.clip(np.round(l_out_float / t_in_scale + t_in_zp), -128, 127).astype(np.int8)
                    
                    common.set_input(tcn_int, t_in_q); tcn_int.invoke()
                    
                    t_raw = np.squeeze(tcn_int.get_tensor(t_out_idx))
                    raw_score = (float(t_raw) - t_out_zp) * t_out_scale
                    
                    smooth_score = smooth_score * 0.3 + raw_score * 0.7
                    fall_score = smooth_score
                    
                    if fall_score > 0.35:
                        status = "FALL DETECTED!"
                    else:
                        status = "NORMAL"

                    nose_y = lpn_in[1] 
                    ankle_l_y, ankle_r_y = lpn_in[23], lpn_in[25] 
                    avg_ankle_y = (ankle_l_y + ankle_r_y) / 2.0
                    
                    log_msg = (f"[Frame {frame_idx:03d}] Score: {fall_score:.4f} | Status: {status:14s} | "
                               f"MaxSpan: {max_span:5.1f}px | Scale: {scale:.4f} | "
                               f"Nose(Y): {nose_y:5.2f} | Ankle(Y): {avg_ankle_y:5.2f}")
                    
                    print(log_msg)
                    txt_log.write(log_msg + '\n') 

    if not is_valid_person: ghost_frames += 1

    if ghost_frames > MAX_GHOST: 
        joint_buffer.clear()
        prev_kps_pixel, prev_scale, smooth_score = None, None, 0.0
        status = "NORMAL"
        draw_kps, confs_out = [], []
    elif len(joint_buffer) > 0 and ghost_frames > 0: 
        joint_buffer.append(joint_buffer[-1])

    curr_time = time.perf_counter()
    instant_fps = 1.0 / (curr_time - last_time + 1e-6)
    last_time = curr_time
    avg_fps = avg_fps * (1 - ALPHA) + instant_fps * ALPHA

    if len(draw_kps) == 13:
        for e in NEW_EDGES:
            p1, p2 = e
            if confs_out[p1] > 0.2 and confs_out[p2] > 0.2:
                cv2.line(frame, (int(draw_kps[p1][0]), int(draw_kps[p1][1])), (int(draw_kps[p2][0]), int(draw_kps[p2][1])), (0, 255, 0), 2)
        for i, (x, y) in enumerate(draw_kps):
            if confs_out[i] > 0.2: 
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    color = (0, 0, 255) if "FALL" in status else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (W, 75), (0, 0, 0), -1)
    info = f"Processing FPS: {avg_fps:.1f} | Frame: {frame_idx}/{total_frames}"
    cv2.putText(frame, info, (10, 25), 1, 1.2, (255, 255, 255), 1)
    cv2.putText(frame, f"STATE: {status} ({fall_score:.2f})", (10, 60), 1, 1.8, color, 2)
    
    out.write(frame)

cap.release()
out.release()
txt_log.close()

print("-" * 80)
print(f"✅ 처리가 완료되었습니다. 비디오: {OUTPUT_VIDEO} / 로그 파일: {LOG_FILE}")