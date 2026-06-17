from ultralytics import YOLO

if __name__ == '__main__': 
    # 1. 아키텍처: 수정한 ReLU6 및 정적 Head 구조 적용
    # 가중치: 사전 학습된 가중치를 로드하되, 초반 충격을 줄입니다.
    model = YOLO('./AI/Project-Code/yolov8n-pose-relu6.yaml').load('yolov8n-pose.pt')

    # 2. 전처리된 낙상 데이터셋으로 학습 시작
    model.train(
        data='C:/Users/pcroom2/Desktop/ASK/AI/dataset/AIP.v2i.yolov8/data.yaml', 
        epochs=100,            
        patience=20,           
        imgsz=224,             
        batch=16,
        device=0,              
        name='v8n_pose_fall_relu_fixed',
        # 🔥 [핵심 추가] 학습 붕괴(0.0002 점수)를 막기 위한 옵션들 🔥
        lr0=0.001,             # 기본값(0.01)보다 10배 낮춰서 천천히 적응시킴
        lrf=0.01,              # 최종 학습률도 더 낮게 유지
        warmup_epochs=5        # 초반 5 에포크 동안은 아주 살살 학습시키며 뇌를 깨움
    )