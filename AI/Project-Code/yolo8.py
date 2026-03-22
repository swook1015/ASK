from ultralytics import YOLO

# 1. 다운로드한 모델 로드
model = YOLO('yolov8n-pose.pt')

# 2. Edge TPU용으로 변환 실행
model.export(format='edgetpu')