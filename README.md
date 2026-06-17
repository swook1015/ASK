# 🛡️ Lightweight 3D Fall Detection System on Edge AI

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Edge TPU](https://img.shields.io/badge/Edge_TPU-Coral-green?style=for-the-badge)](https://coral.ai/)

</div>

## 📝 프로젝트 소개
본 연구는 **Edge TPU(Google Coral)** 환경에서 독립적으로 구동되는 **경량화 3D 낙상 감지 시스템**입니다. `YOLOv8`(2D Pose) → `Lifting Network(LN)`(3D Pose) → `1D-CNN`(Fall Classification)으로 이어지는 파이프라인을 구축하여, 클라우드 의존성 없이 단일 기기에서 실시간 낙상 감지를 수행합니다.

---

## 🚀 실험 결과 요약

### 1. End-to-End 통합 파이프라인 실증 (영상 기반)
원본 영상(.avi)을 파이프라인에 직접 투입하여 실시간 통합 구동 성능을 검증하였습니다.
* **통합 정확도 (Accuracy):** **98.56%**
* **실시간 처리 속도:** **73.5ms/frame** (YOLO+LN+1D-CNN 통합)

### 2. 1D-CNN 분류기 단일 성능 (정답지 기반)
NTU RGB+D Cross-View 프로토콜을 적용하여 모델 고유의 분류 성능을 엄격히 평가하였습니다.
* **Accuracy:** **99.79%**

---

## 🏗️ 시스템 아키텍처 및 최적화
<img src="https://github.com/user-attachments/assets/b4ca7912-91bd-4478-9d4a-a318c5978b4c" alt="이미지 설명" width="1251" height="290"/>

* **연산 효율성**: 16개 관절 대신 핵심 **13개 관절**만을 선별하여 연산량 최적화.
* **경량화 모델**: 전체 파이프라인 약 **2.4MB** 수준의 초경량 구조 설계.
* **양자화 전략**: `INT8` 양자화 적용 시 1D-CNN 성능 열화 **0%** 달성.

---

## 📊 Comparison with SOTA (Accuracy)

| 방법 (Method) | 입력 데이터 | 정확도 (Accuracy) |
| :--- | :---: | :---: |
| 2D Pose + 2D CNN | RGB | 91.70% |
| 3D Pose + LSTM | Depth | 96.12% |
| **본 연구 (1D-CNN)** | **RGB** | **98.56%** |
| Chen et al. (PC/GPU) | RGB | 99.83% |

* 본 연구는 고사양 GPU 환경(Chen et al.) 대비 1.27%p 차이로 근접한 성능을 달성하면서도, 엣지 디바이스에서 실시간 처리가 가능함을 입증함.

---

## 🏃 실행 방법
```bash
# 1. 의존성 설치
프로젝트 구동에 필요한 라이브러리들은 `requirements.txt` 파일에 정의되어 있습니다. 해당 파일은 시스템의 안정적인 실행을 위해 필수적인 패키지 버전 정보를 담고 있는 **의존성(Dependency) 파일**입니다.

# 2. 엣지 가속 환경 최적화
export TF_ENABLE_ONEDNN_OPTS=0
export TF_LITE_DISABLE_XNNPACK=1

# 3. End-to-End 전체 파이프라인 검증 실행
python ALL_PipeLineTest12-0408.py

```
## 연구 및 개발
허성욱 (swook8440@gmail.com)
