import cv2 as cv
import numpy as np
import os
import glob

# =====================================================================
# Camera Calibration
# 체스보드 패턴을 이용한 카메라 캘리브레이션 프로그램
#
# [두 가지 사용 방법]
#
#   방법 A. 미리 찍은 사진 사용 (핸드폰 등으로 미리 촬영한 경우) ← 추천
#     1. calib_images/ 폴더에 사진(jpg/png)을 넣기
#     2. python camera_calibration.py 실행
#     → 폴더에 이미지가 있으면 자동으로 읽어서 캘리브레이션 실행
#
#   방법 B. 실시간 웹캠 캡처
#     1. calib_images/ 폴더를 비워두거나 폴더 없이 실행
#     2. python camera_calibration.py 실행
#     → Space: 캡처 / Enter: 캘리브레이션 / ESC: 종료
#
# [자동 생성 파일/폴더]
#   calib_images/       ← 캡처 이미지 저장 (방법 B)
#   results/K.npy       ← 캘리브레이션 결과 (카메라 행렬)
#   results/dist.npy    ← 캘리브레이션 결과 (왜곡 계수)
# =====================================================================

# ── 설정값 ──────────────────────────────────────────────────────────
BOARD_PATTERN   = (9, 6)           # 체스보드 내부 코너 수 (가로, 세로)
BOARD_CELL_SIZE = 0.025            # 체스보드 한 칸 크기 [m] (A4 출력 기준 약 2.5cm)
MIN_IMAGES      = 10               # 캘리브레이션에 필요한 최소 이미지 수
CALIB_IMG_DIR   = 'calib_images'   # 캡처/입력 이미지 폴더
RESULTS_DIR     = 'results'        # 캘리브레이션 결과 저장 폴더
IMG_EXTENSIONS  = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
# ────────────────────────────────────────────────────────────────────


def load_existing_images():
    """calib_images/ 폴더에서 이미지 파일 목록을 반환"""
    if not os.path.exists(CALIB_IMG_DIR):
        return []
    files = []
    for ext in IMG_EXTENSIONS:
        files += glob.glob(os.path.join(CALIB_IMG_DIR, ext))
    return sorted(files)


def run_calibration_from_images(image_files):
    """
    방법 A: 미리 찍은 이미지 파일들로 캘리브레이션
    """
    print(f'\n[방법 A] calib_images/ 폴더에서 {len(image_files)}장 이미지 로드')

    # 3D 체스보드 기준점 준비 (Z=0 평면)
    obj_pts = np.array(
        [[c, r, 0] for r in range(BOARD_PATTERN[1])
                   for c in range(BOARD_PATTERN[0])],
        dtype=np.float32
    ) * BOARD_CELL_SIZE

    obj_points = []
    img_points = []
    img_size   = None
    failed     = []

    for i, fpath in enumerate(image_files):
        img  = cv.imread(fpath)
        if img is None:
            print(f'  [{i+1:2d}] ❌ 읽기 실패 → {fpath}')
            failed.append(fpath)
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])   # (width, height)

        found, corners = cv.findChessboardCorners(gray, BOARD_PATTERN)
        if found:
            corners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(obj_pts)
            img_points.append(corners)
            print(f'  [{i+1:2d}] ✅ 코너 검출 성공 → {os.path.basename(fpath)}')
        else:
            print(f'  [{i+1:2d}] ⚠️  코너 검출 실패 (체스보드 미검출) → {os.path.basename(fpath)}')
            failed.append(fpath)

    print(f'\n  성공: {len(obj_points)}장 / 전체: {len(image_files)}장')

    if failed:
        print(f'  실패 목록: {[os.path.basename(f) for f in failed]}')

    if len(obj_points) < MIN_IMAGES:
        print(f'\n❌ 오류: 유효한 이미지가 {len(obj_points)}장 뿐입니다.')
        print(f'   최소 {MIN_IMAGES}장 필요. 사진을 더 추가하거나 촬영 조건을 확인하세요.')
        return

    return run_calibration(obj_points, img_points, img_size)


def run_calibration_from_webcam():
    """
    방법 B: 실시간 웹캠으로 캡처하며 캘리브레이션
    """
    print('\n[방법 B] 웹캠 캡처 모드')
    print(f'  Space: 캡처 / Enter: 캘리브레이션 실행 / ESC: 종료\n')

    os.makedirs(CALIB_IMG_DIR, exist_ok=True)

    obj_pts = np.array(
        [[c, r, 0] for r in range(BOARD_PATTERN[1])
                   for c in range(BOARD_PATTERN[0])],
        dtype=np.float32
    ) * BOARD_CELL_SIZE

    obj_points = []
    img_points = []
    img_size   = None

    cap = cv.VideoCapture(0)
    assert cap.isOpened(), '카메라를 열 수 없습니다.'

    board_found = False
    corners     = None
    msg         = ''
    msg_timer   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])

        board_found, corners = cv.findChessboardCorners(gray, BOARD_PATTERN)

        display = frame.copy()
        if board_found:
            corners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv.drawChessboardCorners(display, BOARD_PATTERN, corners, board_found)

        # UI 표시
        h, w = display.shape[:2]
        overlay = display.copy()
        cv.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        status_color = (0, 255, 0) if board_found else (0, 0, 255)
        status_text  = 'Board: FOUND ✔' if board_found else 'Board: NOT FOUND'
        cv.putText(display, status_text, (10, 28),
                   cv.FONT_HERSHEY_DUPLEX, 0.75, status_color, 1)

        progress_color = (0, 255, 0) if len(img_points) >= MIN_IMAGES else (255, 255, 255)
        cv.putText(display, f'Captured: {len(img_points)} / {MIN_IMAGES} (minimum)',
                   (10, 60), cv.FONT_HERSHEY_DUPLEX, 0.65, progress_color, 1)
        cv.putText(display, '[Space] Capture   [Enter] Calibrate   [ESC] Quit',
                   (10, h - 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1)

        if msg and msg_timer > 0:
            msg_timer -= 1
            (tw, th), _ = cv.getTextSize(msg, cv.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cx = (w - tw) // 2
            cy = h // 2
            cv.rectangle(display, (cx-10, cy-th-8), (cx+tw+10, cy+8), (0,0,0), -1)
            cv.putText(display, msg, (cx, cy),
                       cv.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

        cv.imshow('Camera Calibration (Webcam Mode)', display)

        key = cv.waitKey(10) & 0xFF

        if key == ord(' '):
            if board_found:
                obj_points.append(obj_pts)
                img_points.append(corners)
                fname = os.path.join(CALIB_IMG_DIR, f'calib_{len(img_points):03d}.jpg')
                cv.imwrite(fname, frame)
                msg, msg_timer = f'Captured! ({len(img_points)} images)', 35
                print(f'  [{len(img_points):2d}장] 저장 → {fname}')
            else:
                msg, msg_timer = 'Board not found! Adjust angle.', 30

        elif key == 13:   # Enter
            if len(img_points) < MIN_IMAGES:
                need = MIN_IMAGES - len(img_points)
                msg, msg_timer = f'Need {need} more image(s)!', 40
                print(f'  이미지 부족: {need}장 더 캡처하세요.')
            else:
                cap.release()
                cv.destroyAllWindows()
                return run_calibration(obj_points, img_points, img_size)

        elif key == 27:   # ESC
            print('\n프로그램을 종료합니다.')
            break

    cap.release()
    cv.destroyAllWindows()


def run_calibration(obj_points, img_points, img_size):
    """캘리브레이션 실행"""
    print('\n[캘리브레이션 실행 중... 잠시 기다려주세요]')
    rms, K, dist, _, _ = cv.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    save_results(rms, K, dist)


def save_results(rms, K, dist):
    """결과 저장 및 출력"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, 'K.npy'),    K)
    np.save(os.path.join(RESULTS_DIR, 'dist.npy'), dist)

    print('\n' + '=' * 55)
    print('  📷 Camera Calibration Results')
    print('=' * 55)
    print(f'  RMS Error  : {rms:.6f} px')
    print(f'  fx         : {K[0,0]:.4f}')
    print(f'  fy         : {K[1,1]:.4f}')
    print(f'  cx         : {K[0,2]:.4f}')
    print(f'  cy         : {K[1,2]:.4f}')
    print(f'  k1         : {dist[0,0]:.6f}')
    print(f'  k2         : {dist[0,1]:.6f}')
    print(f'  p1         : {dist[0,2]:.6f}')
    print(f'  p2         : {dist[0,3]:.6f}')
    print(f'  k3         : {dist[0,4]:.6f}')

    if   rms < 0.5: quality = '매우 우수 ✅'
    elif rms < 1.0: quality = '우수 ✅'
    elif rms < 2.0: quality = '보통 ⚠️  (다양한 각도 추가 권장)'
    else:           quality = '불량 ❌ (재촬영 필요)'
    print(f'  품질 평가  : {quality}')
    print('=' * 55)
    print(f'\n  결과 저장 완료:')
    print(f'    {RESULTS_DIR}/K.npy')
    print(f'    {RESULTS_DIR}/dist.npy')
    print(f'  → distortion_correction.py 를 실행하세요!\n')


def main():
    print('=' * 55)
    print('  📷 Camera Calibration 시작')
    print('=' * 55)
    print(f'  체스보드: {BOARD_PATTERN[0]}x{BOARD_PATTERN[1]} 내부 코너')
    print(f'  최소 이미지: {MIN_IMAGES}장 (20~30장 권장)')

    # ── 모드 자동 선택 ───────────────────────────────────────────────
    existing_images = load_existing_images()

    if existing_images:
        # 방법 A: calib_images/ 에 사진이 있으면 바로 사용
        print(f'\n  calib_images/ 에서 {len(existing_images)}장 발견 → 방법 A 실행')
        run_calibration_from_images(existing_images)
    else:
        # 방법 B: 사진 없으면 웹캠 캡처 모드
        print(f'\n  calib_images/ 가 비어있음 → 방법 B (웹캠 캡처 모드) 실행')
        run_calibration_from_webcam()


if __name__ == '__main__':
    main()