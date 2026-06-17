ASK (AI-based Real-time Action Recognition System)

본 프로젝트는 한국정보처리학회(KIPS) 컨퍼런스에서 회장상을 수상한 실시간 동작 인식 알고리즘을 구현한 저장소입니다. 엣지 디바이스 환경에서의 동작 인식 파이프라인 최적화 연구를 포함하고 있습니다.

1. 프로젝트 개요

목표: Edge AI 환경에서의 고성능 실시간 3D 동작 인식 파이프라인 구현

핵심 기술: 1D-TCN(Temporal Convolutional Network), Lifting Networks, YOLO 기반 포즈 추정

연구 성과: [ASK 2026] 한국정보처리학회 회장상 수상

2. 주요 코드 경로

모든 핵심 소스 코드는 다음 경로에서 관리됩니다.

경로: AI/Project-Code/01_FinalCode/

3. 기술 성능

성능 지표: 실시간 추론 최적화를 통해 엣지 디바이스에서 동작 인식 정확도와 처리 속도를 극대화하였습니다.

주요 최적화: * 1D-TCN 아키텍처 개선을 통한 추론 효율성 향상

모델 가중치 최적화를 통한 경량화 완료

4. 디렉토리 구조

ASK/
├── AI/
│   ├── Project-Code/
│   │   └── 01_FinalCode/    # 핵심 소스 코드 및 파이프라인
│   ├── dataset/             # (Git 관리 제외 대상)
│   └── models/              # 학습된 모델 가중치
├── requirements.txt         # 의존성 라이브러리 명세
└── README.md


5. 시작하기

# 의존성 설치
pip install -r requirements.txt

# 핵심 파이프라인 실행
cd AI/Project-Code/01_FinalCode/
python [파일명].py


6. 라이선스 및 참조

본 연구는 한국정보처리학회(KIPS) 컨퍼런스 발표 논문을 기반으로 작성되었습니다.
