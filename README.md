# 🛡️ Real-time Fall Detection System: End-to-End Edge AI Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Edge TPU](https://img.shields.io/badge/Edge_TPU-Coral-green?style=for-the-badge)](https://coral.ai/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8--Pose-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)

</div>

## 📝 프로젝트 소개
본 프로젝트는 **Google Coral Edge TPU** 환경에서 독립적으로 구동되는 **고성능 실시간 3D 낙상 감지 파이프라인**입니다. `YOLOv8-Pose`로 관절 좌표를 추출하고, `Lifting Network(LPN)`를 통해 2D 좌표를 3D 공간으로 복원한 뒤, `1D-TCN(Temporal Convolutional Network)` 시계열 분석을 통해 낙상을 판별합니다.

총 **11,000개 이상의 다각도 영상(NTU RGB+D)**을 활용한 통합 실증(End-to-End)을 통해 하드웨어 제약 환경에서도 높은 신뢰성을 입증하였습니다.

---

## 🚀 핵심 성과 및 기능

### 1. 학술 및 실증 지표
* **단일 분류기 성능(모델 검증):** **99.79%** (18,960건의 3D 스켈레톤 정답지 기반)
* **통합 파이프라인 성능(실증):** **98.56%** (11,000건의 Cross-View 원본 영상 기반)
* **실시간 추론 속도:** **73.5ms/frame** (Edge TPU 가속 적용)

### 2. 기술적 핵심 포인트
* **2-Pass 동적 정규화:** 영상 전체의 뼈대 움직임을 사전 분석하여 `Global Max-Dist`를 확보, 카메라 앵글 변화에도 강건한 성능을 보장합니다.
* **시계열 기울기 폭주 방어:** 낙상 판정 임계값(`FALL_THRESHOLD = 0.9`) 최적화 및 `EMA 필터`를 적용하여 일상 동작(물건 줍기 등)에 의한 오탐지(FP)를 원천 차단했습니다.
* **Cold Start 해결:** `Padding` 로직을 통해 영상의 첫 프레임부터 60프레임 버퍼를 확보하여 시스템 기동 직후의 탐지 누락을 방어했습니다.

---

## 🏗️ 시스템 아키텍처

<div align="center">
  <img src="https://via.placeholder.com/800x300?text=YOLOv8-Pose+%E2%86%92+LPN+%E2%86%92+1D-TCN+Pipeline" alt="Architecture" />
</div>

1. **Vision Engine**: YOLOv8-Pose (INT8 Quantized TFLite)
2. **Spatial Recovery**: Lifting Network (LPN) for 3D Reconstruction
3. **Temporal Analysis**: 1D-TCN (8-layer pure NTU)

---

## 📊 Evaluation Results (End-to-End)

| Metric | Performance |
| :--- | :---: |
| **Accuracy** | **99.79%** |
| **Precision** | **96.01%** |
| **Recall** | **91.46%** |
| **F1-Score** | **93.68%** |

> **Confusion Matrix (End-to-End Validation)**
> | | Pred Normal | Pred Fall |
> | :--- | :---: | :---: |
> | **Actual Normal** | 18632 | 12 |
> | **Actual Fall** | 27 | 289 |

---

## 🏃 실행 방법

### 1. 환경 설정
```bash
# 하드웨어 가속 최적화 설정
export TF_ENABLE_ONEDNN_OPTS=0
export TF_LITE_DISABLE_XNNPACK=1
