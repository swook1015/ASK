import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

def letterbox(im, new_shape=(256, 256), color=(114, 114, 114)):
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

# 1. 모델 로드 및 텐서 디테일 추출
POSE_MODEL = 'yolov8n-pose_full_integer_quant_edgetpu.tflite'
LPN_MODEL  = 'lpn_real_seq_60_quant_edgetpu.tflite'
TCN_MODEL  = 'tcn_fall_detector_stride_best.tflite' 

pose_int = make_interpreter(POSE_MODEL); pose_int.allocate_tensors()
lpn_int  = make_interpreter(LPN_MODEL);  lpn_int.allocate_tensors()
tcn_int  = make_interpreter(TCN_MODEL);  tcn_int.allocate_tensors()

# 💡 [핵심] 입력 이미지 양자화 파라미터 추출
p_in_details = pose_int.get_input_details()[0]
in_scale, in_zp = p_in_details['quantization']

p_out_idx = pose_int.get_output_details()[0]['index']
p_scale, p_zp = pose_int.get_output_details()[0]['quantization']
l_out_idx = lpn_int.get_output_details()[0]['index']
t_in_idx, t_out_idx = tcn_int.get_input_details()[0]['index'], tcn_int.get_output_details()[0]['index']
t_scale, t_zp = tcn_int.get_output_details()[0]['quantization']

VIDEO_PATH = '3.mp4'; cap = cv2.VideoCapture(VIDEO_PATH)
v_w, v_h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
out = cv2.VideoWriter('result_final_vision_restored.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (v_w, v_h))

LPN_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NEW_EDGES = [(0,1), (0,2), (1,2), (1,3), (3,5), (2,4), (4,6), (1,7), (2,8), (7,8), (7,9), (9,11), (8,10), (10,12)]

buffer, prev_kps_pixel, status, fall_score = [], None, "NORMAL", 0.0
ghost_frames, MAX_GHOST = 0, 5

print("🚀 [시력 복구 완료] 완벽 추론 파이프라인 가동...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    st = time.perf_counter()
    
    # 1. 이미지 전처리
    img_p, ratio, (pw, ph) = letterbox(frame, (256, 256))
    img_rgb = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
    
    # 💡 [가장 중요] 모델의 시력 복구: float 비율로 변환 후 int8 환경에 맞춰 스케일링
    if p_in_details['dtype'] == np.int8:
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_quantized = np.clip(np.round(img_norm / in_scale + in_zp), -128, 127).astype(np.int8)
        common.set_input(pose_int, img_quantized)
    else:
        common.set_input(pose_int, img_rgb)
        
    pose_int.invoke()
    
    # 2. YOLO 출력 텐서 정리
    raw_out = pose_int.get_tensor(p_out_idx)
    preds_int8 = raw_out[0].transpose(1, 0) if raw_out.shape[1] == 56 else raw_out[0]
    
    preds_float = (preds_int8.astype(np.float32) - p_zp) * p_scale
    
    # 신뢰도 보정 (YOLOv8 모델 특성상 Sigmoid 필요할 수 있음)
    scores = preds_float[:, 4]
    scores = np.where(scores < 0, 1.0 / (1.0 + np.exp(-scores)), scores)
    
    valid_mask = scores > 0.25
    draw_kps, confs_out = [], []

    if np.any(valid_mask):
        valid_preds = preds_float[valid_mask]
        valid_scores = scores[valid_mask]
        
        boxes = np.column_stack((valid_preds[:,0]-valid_preds[:,2]/2, valid_preds[:,1]-valid_preds[:,3]/2, valid_preds[:,2], valid_preds[:,3]))
        nms_idx = pure_numpy_nms(boxes, valid_scores, 0.45)
        
        if len(nms_idx) > 0:
            ghost_frames = 0
            best_row = valid_preds[nms_idx[0]]
            kps = best_row[5:].reshape(17, 3)
            
            # 관절 신뢰도(가시성) Sigmoid 보정
            kps[:, 2] = 1.0 / (1.0 + np.exp(-kps[:, 2]))
            
            # 0~1 비율 텐서일 경우 256픽셀로 복원
            if np.max(kps[:, :2]) <= 1.01:
                kps[:, :2] *= 256.0
                
            target_kps = kps[LPN_INDICES]
            
            # 역산 (화면 원본 크기에 맞춤)
            xs = (target_kps[:, 0] - pw) / ratio
            ys = (target_kps[:, 1] - ph) / ratio
            curr_pixel = np.stack((xs, ys), axis=-1)
            
            prev_kps_pixel = curr_pixel if prev_kps_pixel is None else prev_kps_pixel * 0.4 + curr_pixel * 0.6
            draw_kps = prev_kps_pixel.tolist()
            confs_out = target_kps[:, 2].tolist()

            # 3. LPN 입력 정규화 ([-1, 1] 범위 환산)
            lpn_input_frame = []
            for px, py in prev_kps_pixel:
                lpn_input_frame.extend([(px / v_w) * 2.0 - 1.0, (py / v_h) * 2.0 - 1.0])
                
            buffer.append(lpn_input_frame)
            if len(buffer) > 60: buffer.pop(0)

            # LPN & TCN 낙상 감지
            if len(buffer) == 60:
                seq = np.array(buffer, dtype=np.float32).reshape(1, 60, 26)
                l_s, l_z = lpn_int.get_input_details()[0]['quantization']
                s_q = ((seq / l_s) + l_z).astype(np.int8) if l_s > 0 else seq
                
                common.set_input(lpn_int, s_q)
                lpn_int.invoke()
                tcn_int.set_tensor(t_in_idx, lpn_int.get_tensor(l_out_idx))
                tcn_int.invoke()
                
                fall_score = (float(tcn_int.get_tensor(t_out_idx)[0]) - t_zp) * t_scale
                aspect_ratio = best_row[2] / best_row[3]
                status = "FALL DETECTED!" if fall_score > (0.65 if aspect_ratio > 1.2 else 0.85) else "NORMAL"
        else: ghost_frames += 1
    else: ghost_frames += 1

    if ghost_frames > MAX_GHOST:
        buffer.clear(); prev_kps_pixel = None; status = "NORMAL"
    elif len(buffer) > 0 and ghost_frames > 0:
        buffer.append(buffer[-1])

    # 4. 시각화
    vis_frame = frame.copy()
    if len(draw_kps) == 13:
        for p1, p2 in NEW_EDGES:
            if confs_out[p1] > 0.1 and confs_out[p2] > 0.1:
                cv2.line(vis_frame, (int(draw_kps[p1][0]), int(draw_kps[p1][1])), 
                         (int(draw_kps[p2][0]), int(draw_kps[p2][1])), (0, 255, 0), 2)
        for i, (x, y) in enumerate(draw_kps):
            if confs_out[i] > 0.1:
                cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    curr_fps = 1.0 / (time.perf_counter() - st)
    color = (0, 0, 255) if "FALL" in status else (255, 0, 0)
    cv2.putText(vis_frame, f"{status} {fall_score*100:.1f}% FPS:{curr_fps:.1f}", (30, 50), 1, 1.5, color, 2)
    out.write(vis_frame)

cap.release(); out.release()
print("✅ 시력 복구 및 최종 영상 생성 완료: result_final_vision_restored.avi")