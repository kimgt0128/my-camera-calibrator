# Camera Calibration & Lens Distortion Correction

> **Description**: A camera calibration tool and lens distortion correction program using OpenCV. Calibrates intrinsic camera parameters from a chessboard pattern and applies real-time distortion correction.

---

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 **OpenCV**를 활용하여 체스보드 패턴으로 카메라를 캘리브레이션하고, 렌즈 왜곡을 실시간으로 보정하는 프로그램입니다.

---

## 2. 주요 기능 (Key Features)

### camera_calibration.py
- 웹캠으로 체스보드 패턴을 실시간 촬영하며 코너점 자동 검출
- `Space` 키로 원하는 순간의 프레임을 캡처하여 캘리브레이션 데이터 수집
- `Enter` 키로 캘리브레이션 실행 → `K.npy`, `dist.npy` 자동 저장
- RMS 재투영 오차로 캘리브레이션 품질 자동 평가

### distortion_correction.py
- `camera_calibration.py`로 구한 K, dist_coeff를 바탕으로 실시간 왜곡 보정
- `Space` 키로 보정 전/후를 즉시 비교 가능
- `s` 키로 현재 화면 스크린샷 저장

---

## 3. 조작 가이드 (Controls)

### camera_calibration.py

| 키 (Key) | 설명 (Function) |
|----------|----------------|
| **Space** | 현재 프레임 캡처 (체스보드 검출 시에만) |
| **Enter** | 캘리브레이션 실행 (최소 10장 이상 필요) |
| **ESC** | 프로그램 종료 |

### distortion_correction.py

| 키 (Key) | 설명 (Function) |
|----------|----------------|
| **Space** | 보정 ON/OFF 토글 |
| **s** | 현재 화면 스크린샷 저장 |
| **ESC** | 프로그램 종료 |

---

## 4. 실행 방법 (Setup & Run)

```bash
# 필수 라이브러리 설치
pip install numpy opencv-python

# Step 1: 카메라 캘리브레이션
python camera_calibration.py

# Step 2: 렌즈 왜곡 보정
python distortion_correction.py
```

---

## 5. 캘리브레이션 결과 (Calibration Results)

<!-- 아래 값을 camera_calibration.py 실행 결과로 채워주세요 -->

| 파라미터 | 값 |
|---------|-----|
| **fx** | 000.0000 |
| **fy** | 000.0000 |
| **cx** | 000.0000 |
| **cy** | 000.0000 |
| **k1** | 0.000000 |
| **k2** | 0.000000 |
| **p1** | 0.000000 |
| **p2** | 0.000000 |
| **k3** | 0.000000 |
| **RMS Error** | 0.000000 px |

---

## 6. 왜곡 보정 결과 (Distortion Correction Demo)

<!-- 스크린샷 또는 GIF를 여기에 추가하세요 -->

### 보정 전 (Original)
![original](screenshots/screenshot_001_original.jpg)

### 보정 후 (Corrected)
![corrected](screenshots/screenshot_001_corrected.jpg)