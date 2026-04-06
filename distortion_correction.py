import cv2 as cv
import numpy as np
import os

# =====================================================================
# Lens Distortion Correction
# camera_calibration.py로 구한 K, dist_coeff를 이용한 렌즈 왜곡 보정 프로그램
#
# [조작 방법]
#   Space : 보정 ON/OFF 토글 (비교 모드)
#   s     : 현재 프레임 스크린샷 저장
#   ESC   : 프로그램 종료
# =====================================================================

# ── 설정값 ──────────────────────────────────────────────────────────
K_PATH    = 'K.npy'     # camera_calibration.py 가 저장한 파일
DIST_PATH = 'dist.npy'  # camera_calibration.py 가 저장한 파일
SAVE_DIR  = 'screenshots'
# ────────────────────────────────────────────────────────────────────


def load_camera_params():
    """캘리브레이션 결과 파일 로드"""
    assert os.path.exists(K_PATH),    f'{K_PATH} 파일이 없습니다. camera_calibration.py를 먼저 실행하세요.'
    assert os.path.exists(DIST_PATH), f'{DIST_PATH} 파일이 없습니다. camera_calibration.py를 먼저 실행하세요.'

    K    = np.load(K_PATH)
    dist = np.load(DIST_PATH)

    print('[카메라 파라미터 로드 완료]')
    print(f'  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}')
    print(f'  dist={dist.flatten()}')
    return K, dist


def draw_ui(img, show_corrected, n_saved):
    """UI 오버레이"""
    h, w = img.shape[:2]

    # 상단 상태바
    overlay = img.copy()
    cv.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # 보정 상태
    mode_text  = 'Mode: CORRECTED' if show_corrected else 'Mode: ORIGINAL'
    mode_color = (0, 255, 0)        if show_corrected else (0, 100, 255)
    cv.putText(img, mode_text, (10, 28),
               cv.FONT_HERSHEY_DUPLEX, 0.8, mode_color, 1)

    # 저장 수
    cv.putText(img, f'Saved: {n_saved}', (10, 58),
               cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # 하단 조작 안내
    guide = '[Space] Toggle ON/OFF  [s] Screenshot  [ESC] Quit'
    cv.putText(img, guide, (10, h - 12),
               cv.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

    return img


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 캘리브레이션 파라미터 로드
    K, dist = load_camera_params()

    # 카메라 열기
    cap = cv.VideoCapture(0)
    assert cap.isOpened(), '카메라를 열 수 없습니다.'

    show_corrected = True   # 기본: 보정 ON
    map1, map2     = None, None
    n_saved        = 0

    print('\n[왜곡 보정 시작]')
    print('  Space: 보정 ON/OFF / s: 스크린샷 저장 / ESC: 종료\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── 왜곡 보정 맵 초기화 (최초 1회만) ────────────────────
        if map1 is None:
            map1, map2 = cv.initUndistortRectifyMap(
                K, dist, None, None,
                (w, h),        # ← (width, height) 순서 주의!
                cv.CV_32FC1
            )
            print(f'  언디스토션 맵 생성 완료: {w}x{h}')

        # ── 보정 적용 ────────────────────────────────────────────
        if show_corrected:
            output = cv.remap(frame, map1, map2,
                              interpolation=cv.INTER_LINEAR,
                              borderMode=cv.BORDER_CONSTANT)
        else:
            output = frame.copy()

        # UI 표시
        draw_ui(output, show_corrected, n_saved)
        cv.imshow('Lens Distortion Correction', output)

        key = cv.waitKey(10) & 0xFF

        # ── Space: 보정 ON/OFF 토글 ──────────────────────────────
        if key == ord(' '):
            show_corrected = not show_corrected
            state = 'ON (보정됨)' if show_corrected else 'OFF (원본)'
            print(f'  보정 {state}')

        # ── s: 스크린샷 저장 ─────────────────────────────────────
        elif key == ord('s'):
            n_saved += 1
            state = 'corrected' if show_corrected else 'original'
            fname = os.path.join(SAVE_DIR, f'screenshot_{n_saved:03d}_{state}.jpg')
            cv.imwrite(fname, output)
            print(f'  스크린샷 저장 → {fname}')

        # ── ESC: 종료 ────────────────────────────────────────────
        elif key == 27:
            print('\n프로그램을 종료합니다.')
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()