from ultralytics import YOLO
import numpy as np

yolo_model = YOLO('./AI/models/yolo/yolo26n-pose_full_integer_quant.tflite')
print("로드 완료")
result = yolo_model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=True)
print("추론 완료")
print(result)