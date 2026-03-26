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
POSE_MODEL  = 'yolov8n-pose_full_integer_quant_edgetpu.tflite'
LPN_MODEL   = 'lpn_seq_60_quant_edgetpu.tflite'
TCN_MODEL   = 'tcn_manual_fixed_edgetpu.tflite'

W_IN, H_IN = 512, 512

LPN_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NEW_EDGES = [(0,1),(0,2),(1,2),(1,3),(3,5),(2,4),(4,6),
             (1,7),(2,8),(7,8),(7,9),(9,11),(8,10),(10,12)]

# ==========================================
# 🔧 개선된 하이퍼파라미터
# ==========================================
SCORE_THRESH = 0.25
AVG_CONF_THRESH = 0.15
CORE_CONF_THRESH = 0.2
MIN_AREA = 500 #면적 필터 1500 or 500
KPS_CONF_DRAW = 0.15
NMS_THRESH = 0.5

# ==========================================
# 🚀 [수정 1] 스무딩 파라미터 - 의미 명확화
# ==========================================
# 스무딩 계수가 클수록 현재 프레임을 더 많이 반영
# prev * (1 - smooth) + curr * smooth
SMOOTH_FAST_MOVEMENT = 0.85   # 빠른 움직임: 현재 위치 85% 반영 (즉시 따라감)
SMOOTH_NORMAL = 0.5           # 일반 움직임: 50:50 균형
SMOOTH_STATIONARY = 0.2       # 정지 상태: 과거 80% 유지 (떨림 방지)

# 움직임 감지 임계값 (픽셀 단위)
MOVEMENT_FAST_THRESH = 55     # 이 이상이면 빠른 움직임
MOVEMENT_SLOW_THRESH = 15     # 이 이하면 정지 상태

def letterbox(im, new_shape=(512, 512), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top,    bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left,   right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def pure_numpy_nms(boxes, scores, threshold=0.5):
    if len(boxes) == 0: return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(ovr <= threshold)[0] + 1]
    return keep

def fast_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

# ==========================================
# 🚀 [수정 2] 적응형 스무딩 함수 - 로직 정상화
# ==========================================
def calculate_adaptive_smooth(prev_kps, curr_kps):
    """
    움직임 크기에 따라 스무딩 계수 조절
    
    핵심 원리:
    - 빠른 움직임 → 높은 계수 → 현재 위치 즉시 반영 (스켈레톤이 따라감)
    - 느린 움직임 → 낮은 계수 → 과거 위치 유지 (노이즈/떨림 방지)
    """
    if prev_kps is None:
        return SMOOTH_NORMAL
    
    # 프레임 간 평균 이동 거리 계산 (픽셀 단위)
    movement = np.mean(np.linalg.norm(curr_kps - prev_kps, axis=1))
    
    # 🔥 수정된 로직: 움직임이 클수록 현재 프레임을 강하게 반영
    if movement > MOVEMENT_FAST_THRESH:
        # 빠른 움직임 (낙상, 빠른 허리 숙임 등)
        # 현재 위치를 85% 반영하여 즉각적으로 따라감
        return SMOOTH_FAST_MOVEMENT
    elif movement > MOVEMENT_SLOW_THRESH:
        # 일상적인 움직임 (걷기, 천천히 움직이기)
        return SMOOTH_NORMAL
    else:
        # 정지 또는 미세한 움직임
        # 과거 위치를 80% 유지하여 감지 노이즈로 인한 떨림 방지
        return SMOOTH_STATIONARY

# ==========================================
# ⌨️ 2. 사용자 입력 처리
# ==========================================
print("=" * 60)
print("🎥 낙상 감지 v2.1 (스무딩 로직 정상화) 🎥")
print("=" * 60)

while True:
    try:
        user_input = input("▶ 영상 파일 이름(확장자 제외)을 입력하세요 (종료: q) : ").strip()
        if user_input.lower() == 'q':
            print("프로그램을 종료합니다."); sys.exit(0)
        INPUT_VIDEO = f"{user_input}.mp4"
        if not os.path.exists(INPUT_VIDEO):
            print(f"🚨 에러: '{INPUT_VIDEO}' 파일이 없습니다.\n"); continue
        break
    except KeyboardInterrupt:
        print("\n강제 종료."); sys.exit(0)

OUTPUT_VIDEO = f'{INPUT_VIDEO}_result.avi'
LOG_FILE     = f'{INPUT_VIDEO}_log.txt'

# ==========================================
# 🧠 3. AI 모델 로드
# ==========================================
print("\n🚀 AI 모델 로드 중...")
pose_int = make_interpreter(POSE_MODEL); pose_int.allocate_tensors()
lpn_int  = make_interpreter(LPN_MODEL);  lpn_int.allocate_tensors()
tcn_int  = make_interpreter(TCN_MODEL);  tcn_int.allocate_tensors()

p_in_details  = pose_int.get_input_details()[0]
in_scale, in_zp = p_in_details['quantization']
p_out_details = pose_int.get_output_details()[0]
p_out_idx     = p_out_details['index']
p_scale, p_zp = p_out_details['quantization']

l_in_scale,  l_in_zp  = lpn_int.get_input_details()[0]['quantization']
l_out_idx              = lpn_int.get_output_details()[0]['index']
l_out_scale, l_out_zp = lpn_int.get_output_details()[0]['quantization']

t_in_idx               = tcn_int.get_input_details()[0]['index']
t_in_scale,  t_in_zp  = tcn_int.get_input_details()[0]['quantization']
t_out_idx              = tcn_int.get_output_details()[0]['index']
t_out_scale, t_out_zp = tcn_int.get_output_details()[0]['quantization']

cap = cv2.VideoCapture(INPUT_VIDEO)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS == 0 or np.isnan(FPS): FPS = 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

txt_log = open(LOG_FILE, 'w', encoding='utf-8')
txt_log.write(f"--- Fall Detection Log v2.1 for {INPUT_VIDEO} ---\n")
txt_log.write(f"Settings: SCORE>{SCORE_THRESH}, AVG_CONF>{AVG_CONF_THRESH}, CORE_CONF>{CORE_CONF_THRESH}, AREA>{MIN_AREA}\n")
txt_log.write(f"Smoothing: Fast={SMOOTH_FAST_MOVEMENT}, Normal={SMOOTH_NORMAL}, Static={SMOOTH_STATIONARY}\n")
txt_log.write("-" * 80 + "\n")

joint_buffer    = []
prev_kps_pixel  = None
prev_kps_raw    = None  # 스무딩 전 raw 좌표 (적응형 스무딩용)
prev_scale      = None
fall_score      = 0.0        
status          = "NORMAL"
ghost_frames    = 0
MAX_GHOST       = 30
draw_kps        = []
confs_out       = []
ALPHA           = 0.1
avg_fps         = 0.0
frame_idx       = 0
last_time       = time.perf_counter()

# 디버깅용 카운터
debug_no_person = 0
debug_low_conf  = 0
debug_low_area  = 0
debug_valid     = 0

# ==========================================
# ⚙️ 4. 영상 처리 루프
# ==========================================
print(f"🎬 영상 처리 시작: {W}x{H} @ {FPS}FPS (총 {total_frames} 프레임)")
print(f"📊 임계값: Score>{SCORE_THRESH}, AvgConf>{AVG_CONF_THRESH}, CoreConf>{CORE_CONF_THRESH}, Area>{MIN_AREA}")
print(f"🔄 스무딩: Fast={SMOOTH_FAST_MOVEMENT}, Normal={SMOOTH_NORMAL}, Static={SMOOTH_STATIONARY}")
print("-" * 80)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_idx += 1
    t_start = time.perf_counter()

    img_p, ratio, (pad_left, pad_top) = letterbox(frame, (W_IN, H_IN))
    img_rgb   = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
    img_input = np.clip(
        np.round((img_rgb / 255.0) / in_scale + in_zp), -128, 127
    ).astype(np.int8)

    common.set_input(pose_int, img_input)
    pose_int.invoke()

    raw_out   = pose_int.get_tensor(p_out_idx)
    preds_raw = raw_out[0].transpose(1, 0) if raw_out.shape[1] == 56 else raw_out[0]
    preds_float = (preds_raw.astype(np.float32) - p_zp) * p_scale

    scores     = fast_sigmoid(preds_float[:, 4])
    valid_mask = scores > SCORE_THRESH
    is_valid_person = False
    debug_reason = ""

    if np.any(valid_mask):
        v_preds, v_scores = preds_float[valid_mask], scores[valid_mask]
        v_preds[:, :4] *= float(W_IN)
        boxes   = np.column_stack((
            v_preds[:, 0] - v_preds[:, 2] / 2,
            v_preds[:, 1] - v_preds[:, 3] / 2,
            v_preds[:, 2], v_preds[:, 3]
        ))
        nms_idx = pure_numpy_nms(boxes, v_scores, NMS_THRESH)

        if len(nms_idx) > 0:
            best_row = v_preds[nms_idx[0]]
            best_score = v_scores[nms_idx[0]]
            kps = best_row[5:].reshape(17, 3)
            kps[:, :2] *= float(W_IN)
            kps[:, 2]   = fast_sigmoid(kps[:, 2])

            target_kps = kps[LPN_INDICES]
            confs_out  = target_kps[:, 2].tolist()

            avg_kp_conf = sum(confs_out) / len(confs_out)
            core_conf = (confs_out[1] + confs_out[2] + confs_out[7] + confs_out[8]) / 4.0
            
            bbox_w = np.max(target_kps[:, 0]) - np.min(target_kps[:, 0])
            bbox_h = np.max(target_kps[:, 1]) - np.min(target_kps[:, 1])
            area   = bbox_w * bbox_h

            if avg_kp_conf > AVG_CONF_THRESH and core_conf > CORE_CONF_THRESH and area > MIN_AREA:
                is_valid_person = True
                debug_valid += 1
                ghost_frames = 0

                xs = (target_kps[:, 0] - pad_left) / ratio
                ys = (target_kps[:, 1] - pad_top)  / ratio
                curr_pixel = np.stack((xs, ys), axis=-1)

                # 🚀 [수정됨] 적응형 스무딩 - 이제 올바르게 작동
                adaptive_smooth = calculate_adaptive_smooth(prev_kps_raw, curr_pixel)
                prev_kps_raw = curr_pixel.copy()  # raw 좌표 저장
                
                if prev_kps_pixel is None: 
                    prev_kps_pixel = curr_pixel
                else: 
                    # smooth가 높을수록 curr_pixel을 더 많이 반영
                    prev_kps_pixel = prev_kps_pixel * (1.0 - adaptive_smooth) + curr_pixel * adaptive_smooth

                draw_kps = prev_kps_pixel.tolist()
                root_x, root_y = np.mean(prev_kps_pixel[7:9], axis=0)

                span_x    = np.max(prev_kps_pixel[:, 0]) - np.min(prev_kps_pixel[:, 0])
                span_y    = np.max(prev_kps_pixel[:, 1]) - np.min(prev_kps_pixel[:, 1])
                span_max  = max(span_x, span_y)
                raw_scale = 1.6642 / (span_max + 1e-6)

                if prev_scale is None: prev_scale = raw_scale
                else: prev_scale = prev_scale * 0.5 + raw_scale * 0.5

                lpn_in = []
                for px, py in prev_kps_pixel:
                    lpn_in.extend([
                        (px - root_x) * prev_scale,
                        -(py - root_y) * prev_scale
                    ])

                joint_buffer.append(lpn_in)
                if len(joint_buffer) > 60: joint_buffer.pop(0)

                if len(joint_buffer) == 60:
                    seq = np.array(joint_buffer, dtype=np.float32).reshape(1, 60, 26)
                    s_q = np.clip(np.round(seq / l_in_scale + l_in_zp), -128, 127).astype(np.int8)
                    common.set_input(lpn_int, s_q); lpn_int.invoke()

                    l_out_fp = (lpn_int.get_tensor(l_out_idx).astype(np.float32) - l_out_zp) * l_out_scale

                    t_in_q = np.clip(np.round(l_out_fp / t_in_scale + t_in_zp), -128, 127).astype(np.int8)
                    common.set_input(tcn_int, t_in_q); tcn_int.invoke()

                    t_raw = np.squeeze(tcn_int.get_tensor(t_out_idx))
                    fall_score = (float(t_raw) - t_out_zp) * t_out_scale

                    # 🛡️ 허리 굽힘 방어막 (Bending Guard)
                    if fall_score > 0.4:
                        hip_y = (prev_kps_pixel[7][1] + prev_kps_pixel[8][1]) / 2.0
                        ankle_y = (prev_kps_pixel[11][1] + prev_kps_pixel[12][1]) / 2.0
                        hip_to_ankle = ankle_y - hip_y 
                        
                        aspect_ratio = span_x / (span_y + 1e-6)
                        is_lying = aspect_ratio > 1.2
                        is_bending = (aspect_ratio < 1.5) and (hip_to_ankle > span_y * 0.35)
                        
                        if is_lying:
                            status = "FALL DETECTED!"
                        elif is_bending:
                            status = "BENDING"
                        else:
                            status = "FALL DETECTED!"
                    else:
                        status = "NORMAL"

                    log_msg = (f"[Frame {frame_idx:04d}] Score: {fall_score:.4f} | "
                               f"Status: {status:14s} | "
                               f"SpanXY: {span_x:.0f}x{span_y:.0f}px | Scale: {prev_scale:.4f} | "
                               f"AvgConf: {avg_kp_conf:.2f} | Smooth: {adaptive_smooth:.2f}")
                    print(log_msg)
                    txt_log.write(log_msg + '\n')
                else:
                    debug_msg = f"[Frame {frame_idx:04d}] Buffer: {len(joint_buffer)}/60 | AvgConf: {avg_kp_conf:.2f} | Area: {area:.0f}"
                    print(debug_msg)
            else:
                if avg_kp_conf <= AVG_CONF_THRESH:
                    debug_low_conf += 1
                    debug_reason = f"LowAvgConf({avg_kp_conf:.2f})"
                elif core_conf <= CORE_CONF_THRESH:
                    debug_low_conf += 1
                    debug_reason = f"LowCoreConf({core_conf:.2f})"
                else:
                    debug_low_area += 1
                    debug_reason = f"LowArea({area:.0f})"
        else:
            debug_no_person += 1
            debug_reason = "NMS_empty"
    else:
        debug_no_person += 1
        debug_reason = f"NoDetection(max_score={np.max(scores):.3f})"

    if not is_valid_person: 
        ghost_frames += 1
        if frame_idx % 10 == 0 and debug_reason:
            print(f"[Frame {frame_idx:04d}] ⚠️ {debug_reason} | Ghost: {ghost_frames}")

    if ghost_frames > MAX_GHOST:
        joint_buffer.clear()
        prev_kps_pixel, prev_kps_raw, prev_scale, fall_score = None, None, None, 0.0
        status = "NORMAL"
        draw_kps, confs_out = [], []
    elif len(joint_buffer) > 0 and ghost_frames > 0:
        joint_buffer.append(joint_buffer[-1])

    curr_time   = time.perf_counter()
    instant_fps = 1.0 / (curr_time - last_time + 1e-6)
    last_time   = curr_time
    avg_fps     = avg_fps * (1 - ALPHA) + instant_fps * ALPHA

    # 스켈레톤 그리기
    if len(draw_kps) == 13:
        for p1, p2 in NEW_EDGES:
            if confs_out[p1] > KPS_CONF_DRAW and confs_out[p2] > KPS_CONF_DRAW:
                cv2.line(frame,
                         (int(draw_kps[p1][0]), int(draw_kps[p1][1])),
                         (int(draw_kps[p2][0]), int(draw_kps[p2][1])),
                         (0, 255, 0), 2)
        for i, (x, y) in enumerate(draw_kps):
            if confs_out[i] > KPS_CONF_DRAW:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    if "FALL" in status: color = (0, 0, 255)
    elif "BENDING" in status: color = (0, 255, 255)
    else: color = (0, 255, 0)
    
    cv2.rectangle(frame, (0, 0), (W, 75), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS:{avg_fps:.1f} | Frame:{frame_idx}/{total_frames}",
                (10, 25), 1, 1.2, (255, 255, 255), 1)
    cv2.putText(frame, f"STATE: {status} ({fall_score:.2f})",
                (10, 60), 1, 1.8, color, 2)
    out.write(frame)

cap.release()
out.release()

txt_log.write("-" * 80 + "\n")
txt_log.write(f"총 프레임: {frame_idx}\n")
txt_log.write(f"유효 감지: {debug_valid} ({100*debug_valid/frame_idx:.1f}%)\n")
txt_log.write(f"미감지(NoDetection/NMS): {debug_no_person}\n")
txt_log.write(f"낮은 신뢰도: {debug_low_conf}\n")
txt_log.write(f"작은 영역: {debug_low_area}\n")
txt_log.close()

print("-" * 80)
print(f"📊 통계: 유효감지 {debug_valid}/{frame_idx} ({100*debug_valid/frame_idx:.1f}%)")
print(f"✅ 완료 → 비디오: {OUTPUT_VIDEO} / 로그: {LOG_FILE}")