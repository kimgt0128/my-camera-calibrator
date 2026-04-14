import cv2 as cv
import numpy as np
import os
import glob

# =====================================================================
# Camera Calibration
# 체스보드 패턴을 이용한 카메라 캘리브레이션 프로그램
#
# [세 가지 사용 방법 — 자동 감지]
#
#   방법 A. 동영상에서 프레임 자동 추출 ← 핵심 추가 기능
#     1. videos/ 폴더에 동영상(mp4/mov/avi) 넣기
#     2. python camera_calibration.py 실행
#     → 각 동영상에서 균등하게 프레임 추출
#     → 체스보드 검출된 프레임만 골라 calib_images/ 에 저장
#     → 자동으로 캘리브레이션 실행
#
#   방법 B. 미리 찍은 사진 사용
#     1. calib_images/ 폴더에 사진(jpg/png) 넣기
#     2. python camera_calibration.py 실행
#
#   방법 C. 실시간 웹캠 캡처
#     1. videos/, calib_images/ 모두 비어있거나 없을 때
#     2. python camera_calibration.py 실행
#     → Space: 캡처 / Enter: 캘리브레이션 / ESC: 종료
#
# [자동 생성 파일/폴더]
#   calib_images/       ← 추출/캡처된 체스보드 이미지
#   results/K.npy       ← 카메라 행렬
#   results/dist.npy    ← 왜곡 계수
# =====================================================================

# ── 설정값 ──────────────────────────────────────────────────────────
BOARD_PATTERN    = (9, 6)          # 체스보드 내부 코너 수 (가로, 세로)
BOARD_CELL_SIZE  = 0.025           # 체스보드 한 칸 크기 [m]
MIN_IMAGES       = 10              # 캘리브레이션 최소 이미지 수
TARGET_IMAGES    = 25              # 동영상에서 추출할 목표 이미지 수
VIDEOS_DIR       = 'videos'        # 동영상 입력 폴더
CALIB_IMG_DIR    = 'calib_images'  # 체스보드 이미지 저장 폴더
RESULTS_DIR      = 'results'       # 캘리브레이션 결과 저장 폴더
VIDEO_EXTENSIONS = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV', '*.AVI')
IMG_EXTENSIONS   = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
# ────────────────────────────────────────────────────────────────────


# ── 공통 유틸 ────────────────────────────────────────────────────────

def make_obj_pts():
    """3D 체스보드 기준점 생성 (Z=0 평면)"""
    return np.array(
        [[c, r, 0] for r in range(BOARD_PATTERN[1])
                   for c in range(BOARD_PATTERN[0])],
        dtype=np.float32
    ) * BOARD_CELL_SIZE


def detect_corners(gray):
    """체스보드 코너 검출 + 서브픽셀 정제. 실패 시 None 반환"""
    
    # ── 멀티스케일 검출: 원본 → 1.5x → 2x 순으로 시도 ──────────
    scales = [1.0, 1.5, 2.0]
    
    for scale in scales:
        if scale == 1.0:
            resized = gray
        else:
            h, w = gray.shape
            resized = cv.resize(gray, (int(w*scale), int(h*scale)),
                                interpolation=cv.INTER_LINEAR)
        
        found, corners = cv.findChessboardCorners(
            resized, BOARD_PATTERN,
            # ↓ 추가: 더 적극적으로 탐색하는 플래그
            flags=cv.CALIB_CB_ADAPTIVE_THRESH +
                  cv.CALIB_CB_NORMALIZE_IMAGE +
                  cv.CALIB_CB_FAST_CHECK
        )
        
        if found:
            # 확대된 좌표를 원본 크기로 되돌리기
            if scale != 1.0:
                corners = corners / scale
            
            # 서브픽셀 정제는 원본 이미지 기준으로
            corners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            return corners
    
    return None


def run_calibration(obj_points, img_points, img_size):
    """캘리브레이션 실행 후 결과 저장"""
    print(f'\n[캘리브레이션 실행 중... {len(obj_points)}장 사용]')
    rms, K, dist, _, _ = cv.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    save_results(rms, K, dist)


def save_results(rms, K, dist):
    """결과 저장 및 터미널 출력"""
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
    print(f'\n  결과 저장: {RESULTS_DIR}/K.npy, {RESULTS_DIR}/dist.npy')
    print('  → distortion_correction.py 를 실행하세요!\n')


# ── 방법 A: 동영상에서 프레임 추출 ──────────────────────────────────

def get_rotation_from_exif(video_path):
    """
    핸드폰 동영상은 EXIF 회전 정보가 없어서 OpenCV가 그대로 읽음.
    영상의 가로/세로 비율로 세로 영상 여부를 판단하여
    필요한 회전 방향을 반환.
    """
    cap = cv.VideoCapture(video_path)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if h > w:
        # 세로 영상 (1080x1920 등) → 시계방향 90도 회전
        return cv.ROTATE_90_CLOCKWISE
    return None  # 가로 영상 → 회전 불필요


def rotate_frame(frame, rotation):
    """회전 플래그에 따라 프레임 회전"""
    if rotation is None:
        return frame
    return cv.rotate(frame, rotation)


def extract_frames_from_video(video_path, n_extract):
    """
    동영상에서 n_extract개의 프레임을 균등하게 추출.
    세로 영상(핸드폰 촬영)은 자동으로 가로로 회전하여 반환.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  ❌ 동영상 열기 실패: {os.path.basename(video_path)}')
        return []

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0
    w            = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 세로 영상 감지 및 회전 방향 결정
    rotation     = get_rotation_from_exif(video_path)
    orient_text  = '세로 → 자동 회전 적용' if rotation is not None else '가로 → 회전 없음'

    print(f'  📹 {os.path.basename(video_path)}'
          f' | {total_frames}프레임 | {duration:.1f}초 | {fps:.1f}fps'
          f' | {w}x{h} ({orient_text})')

    # 균등 간격 프레임 인덱스 계산
    if total_frames <= n_extract:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / n_extract) for i in range(n_extract)]

    frames = []
    for idx in indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(rotate_frame(frame, rotation))  # ← 회전 적용

    cap.release()
    return frames


def run_calibration_from_videos(video_files):
    print(f'\n[방법 A] {len(video_files)}개 동영상에서 프레임 추출')
    os.makedirs(CALIB_IMG_DIR, exist_ok=True)

    obj_pts    = make_obj_pts()
    obj_points = []
    img_points = []
    img_size   = None
    saved_count = 0

    for v_idx, vpath in enumerate(video_files):
        print(f'\n  [{v_idx+1}/{len(video_files)}] 처리 중...')
        
        # ↓ 변경: 전체 프레임 다 추출
        frames = extract_frames_from_video(vpath, 9999)

        detected = 0
        for frame in frames:
            gray    = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners = detect_corners(gray)

            if corners is not None:
                if img_size is None:
                    img_size = (frame.shape[1], frame.shape[0])

                obj_points.append(obj_pts)
                img_points.append(corners)

                saved_count += 1
                fname = os.path.join(CALIB_IMG_DIR, f'calib_{saved_count:03d}.jpg')
                cv.imwrite(fname, frame)
                detected += 1

                # ↓ 변경: 동영상당 제한 없앰 (전부 수집)

        print(f'     → 체스보드 검출 성공: {detected}장 / 전체 {len(frames)}프레임')

    print(f'\n  총 유효 이미지: {len(obj_points)}장')

    if len(obj_points) < MIN_IMAGES:
        print(f'\n❌ 유효 이미지 부족: {len(obj_points)}장 (최소 {MIN_IMAGES}장 필요)')
        print('   체스보드가 선명하게 찍힌 부분이 부족합니다.')
        return

    run_calibration(obj_points, img_points, img_size)


# ── 방법 B: 이미지 파일 직접 사용 ───────────────────────────────────

def run_calibration_from_images(image_files):
    """방법 B: calib_images/ 의 이미지 파일로 캘리브레이션"""
    print(f'\n[방법 B] calib_images/ 에서 {len(image_files)}장 이미지 로드')

    obj_pts    = make_obj_pts()
    obj_points = []
    img_points = []
    img_size   = None
    failed     = []

    for i, fpath in enumerate(image_files):
        img = cv.imread(fpath)
        if img is None:
            print(f'  [{i+1:2d}] ❌ 읽기 실패 → {os.path.basename(fpath)}')
            failed.append(fpath)
            continue

        gray    = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners = detect_corners(gray)

        if img_size is None:
            img_size = (img.shape[1], img.shape[0])

        if corners is not None:
            obj_points.append(obj_pts)
            img_points.append(corners)
            print(f'  [{i+1:2d}] ✅ 검출 성공 → {os.path.basename(fpath)}')
        else:
            print(f'  [{i+1:2d}] ⚠️  체스보드 미검출 → {os.path.basename(fpath)}')
            failed.append(fpath)

    print(f'\n  성공: {len(obj_points)}장 / 전체: {len(image_files)}장')

    if len(obj_points) < MIN_IMAGES:
        print(f'\n❌ 유효 이미지 부족: {len(obj_points)}장 (최소 {MIN_IMAGES}장 필요)')
        return

    run_calibration(obj_points, img_points, img_size)


# ── 방법 C: 실시간 웹캠 ──────────────────────────────────────────────

def run_calibration_from_webcam():
    """방법 C: 실시간 웹캠으로 캡처하며 캘리브레이션"""
    print('\n[방법 C] 웹캠 캡처 모드')
    print(f'  Space: 캡처 / Enter: 캘리브레이션 / ESC: 종료\n')
    os.makedirs(CALIB_IMG_DIR, exist_ok=True)

    obj_pts    = make_obj_pts()
    obj_points = []
    img_points = []
    img_size   = None

    cap = cv.VideoCapture(0)
    assert cap.isOpened(), '카메라를 열 수 없습니다.'

    corners   = None
    msg       = ''
    msg_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])

        corners = detect_corners(gray)
        board_found = corners is not None

        display = frame.copy()
        if board_found:
            cv.drawChessboardCorners(display, BOARD_PATTERN, corners, True)

        # UI
        h, w = display.shape[:2]
        ov = display.copy()
        cv.rectangle(ov, (0, 0), (w, 80), (0, 0, 0), -1)
        cv.addWeighted(ov, 0.5, display, 0.5, 0, display)

        sc = (0, 255, 0) if board_found else (0, 0, 255)
        st = 'Board: FOUND ✔' if board_found else 'Board: NOT FOUND'
        cv.putText(display, st, (10, 28), cv.FONT_HERSHEY_DUPLEX, 0.75, sc, 1)

        pc = (0, 255, 0) if len(img_points) >= MIN_IMAGES else (255, 255, 255)
        cv.putText(display, f'Captured: {len(img_points)} / {MIN_IMAGES} (min)',
                   (10, 60), cv.FONT_HERSHEY_DUPLEX, 0.65, pc, 1)
        cv.putText(display, '[Space] Capture  [Enter] Calibrate  [ESC] Quit',
                   (10, h - 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1)

        if msg and msg_timer > 0:
            msg_timer -= 1
            (tw, th), _ = cv.getTextSize(msg, cv.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cx, cy = (w - tw) // 2, h // 2
            cv.rectangle(display, (cx-10, cy-th-8), (cx+tw+10, cy+8), (0,0,0), -1)
            cv.putText(display, msg, (cx, cy), cv.FONT_HERSHEY_DUPLEX, 0.9, (0,255,255), 2)

        cv.imshow('Camera Calibration (Webcam)', display)
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
                msg, msg_timer = 'Board not found!', 30

        elif key == 13:  # Enter
            if len(img_points) < MIN_IMAGES:
                need = MIN_IMAGES - len(img_points)
                msg, msg_timer = f'Need {need} more!', 40
            else:
                cap.release()
                cv.destroyAllWindows()
                run_calibration(obj_points, img_points, img_size)
                return

        elif key == 27:  # ESC
            print('\n종료합니다.')
            break

    cap.release()
    cv.destroyAllWindows()


# ── 메인 ─────────────────────────────────────────────────────────────

def get_files(directory, extensions):
    """폴더에서 확장자에 맞는 파일 목록 반환"""
    if not os.path.exists(directory):
        return []
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(directory, ext))
    return sorted(files)


def main():
    print('=' * 55)
    print('  📷 Camera Calibration 시작')
    print('=' * 55)
    print(f'  체스보드: {BOARD_PATTERN[0]}x{BOARD_PATTERN[1]} 내부 코너')
    print(f'  목표 이미지: {TARGET_IMAGES}장 (최소 {MIN_IMAGES}장)\n')

    video_files = get_files(VIDEOS_DIR,    VIDEO_EXTENSIONS)
    image_files = get_files(CALIB_IMG_DIR, IMG_EXTENSIONS)

    if video_files:
        # 방법 A: 동영상 우선
        print(f'  videos/ 에서 {len(video_files)}개 동영상 발견 → 방법 A 실행')
        run_calibration_from_videos(video_files)

    elif image_files:
        # 방법 B: 이미지 파일
        print(f'  calib_images/ 에서 {len(image_files)}장 발견 → 방법 B 실행')
        run_calibration_from_images(image_files)

    else:
        # 방법 C: 웹캠
        print('  videos/, calib_images/ 모두 비어있음 → 방법 C (웹캠) 실행')
        run_calibration_from_webcam()


if __name__ == '__main__':
    main()