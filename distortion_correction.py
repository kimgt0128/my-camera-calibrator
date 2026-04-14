import cv2 as cv
import numpy as np
import os
import glob

# ── 설정값 ──────────────────────────────────────────────────────────
_BASE     = os.path.dirname(os.path.abspath(__file__))
K_PATH    = os.path.join(_BASE, 'results', 'K.npy')
DIST_PATH = os.path.join(_BASE, 'results', 'dist.npy')
SAVE_DIR  = os.path.join(_BASE, 'screenshots')

# ↓ 이미지 입력 폴더 (여기에 사진 넣으면 이미지 모드로 실행)
INPUT_IMG_DIR = os.path.join(_BASE, 'input_images')
IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
# ────────────────────────────────────────────────────────────────────


def load_camera_params():
    assert os.path.exists(K_PATH),    f'{K_PATH} 파일이 없습니다.'
    assert os.path.exists(DIST_PATH), f'{DIST_PATH} 파일이 없습니다.'
    K    = np.load(K_PATH)
    dist = np.load(DIST_PATH)
    print('[카메라 파라미터 로드 완료]')
    print(f'  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}')
    return K, dist


# ── 이미지 모드 ──────────────────────────────────────────────────────
def run_image_mode(K, dist, image_files):
    print(f'\n[이미지 모드] {len(image_files)}장 처리')
    os.makedirs(SAVE_DIR, exist_ok=True)

    for fpath in image_files:
        img = cv.imread(fpath)
        if img is None:
            print(f'  ❌ 읽기 실패: {os.path.basename(fpath)}')
            continue

        # ↓ 추가: 세로 이미지면 가로로 회전 (캘리브레이션과 동일한 방향)
        if img.shape[0] > img.shape[1]:
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            print(f'  ↩️  세로 이미지 감지 → 90도 회전 적용')

        h, w = img.shape[:2]
        name = os.path.splitext(os.path.basename(fpath))[0]

        # 왜곡 보정 맵 생성
        map1, map2 = cv.initUndistortRectifyMap(
            K, dist, None, None, (w, h), cv.CV_32FC1
        )
        corrected = cv.remap(img, map1, map2,
                             interpolation=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_CONSTANT)

        # 원본 / 보정 나란히 비교 이미지 생성
        compare = np.hstack([img, corrected])

        # 라벨 텍스트
        cv.putText(compare, 'Original',  (20, 40),
                   cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
        cv.putText(compare, 'Corrected', (w + 20, 40),
                   cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

        # 저장
        orig_path    = os.path.join(SAVE_DIR, f'{name}_original.jpg')
        corr_path    = os.path.join(SAVE_DIR, f'{name}_corrected.jpg')
        compare_path = os.path.join(SAVE_DIR, f'{name}_compare.jpg')

        cv.imwrite(orig_path,    img)
        cv.imwrite(corr_path,    corrected)
        cv.imwrite(compare_path, compare)
        print(f'  ✅ {name} → 저장 완료')

        # 화면 표시 (ESC로 넘어가기)
        display = compare.copy()
        if display.shape[1] > 1600:  # 너무 크면 축소
            scale = 1600 / display.shape[1]
            display = cv.resize(display, None, fx=scale, fy=scale)
        cv.imshow(f'Distortion Correction - {name}  [ESC: next]', display)
        cv.waitKey(0)
        cv.destroyAllWindows()

    print(f'\n  결과 저장 위치: {SAVE_DIR}/')


# ── 웹캠 모드 (기존 코드 그대로) ────────────────────────────────────
def draw_ui(img, show_corrected, n_saved):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    mode_text  = 'Mode: CORRECTED' if show_corrected else 'Mode: ORIGINAL'
    mode_color = (0, 255, 0) if show_corrected else (0, 100, 255)
    cv.putText(img, mode_text, (10, 28),  cv.FONT_HERSHEY_DUPLEX, 0.8, mode_color, 1)
    cv.putText(img, f'Saved: {n_saved}', (10, 58), cv.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv.putText(img, '[Space] Toggle  [s] Screenshot  [ESC] Quit',
               (10, h - 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (200,200,200), 1)
    return img


def run_webcam_mode(K, dist):
    cap = cv.VideoCapture(0)
    assert cap.isOpened(), '카메라를 열 수 없습니다.'
    os.makedirs(SAVE_DIR, exist_ok=True)

    show_corrected = True
    map1, map2 = None, None
    n_saved = 0

    print('\n[웹캠 모드]  Space: 보정 ON/OFF / s: 저장 / ESC: 종료\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if map1 is None:
            map1, map2 = cv.initUndistortRectifyMap(
                K, dist, None, None, (w, h), cv.CV_32FC1
            )
            print(f'  언디스토션 맵 생성 완료: {w}x{h}')

        output = cv.remap(frame, map1, map2,
                          interpolation=cv.INTER_LINEAR,
                          borderMode=cv.BORDER_CONSTANT) if show_corrected else frame.copy()

        draw_ui(output, show_corrected, n_saved)
        cv.imshow('Lens Distortion Correction', output)
        key = cv.waitKey(10) & 0xFF

        if key == ord(' '):
            show_corrected = not show_corrected
        elif key == ord('s'):
            n_saved += 1
            state = 'corrected' if show_corrected else 'original'
            fname = os.path.join(SAVE_DIR, f'screenshot_{n_saved:03d}_{state}.jpg')
            cv.imwrite(fname, output)
            print(f'  저장 → {fname}')
        elif key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


# ── 메인: 이미지 폴더 있으면 이미지 모드, 없으면 웹캠 모드 ──────────
def get_image_files(directory):
    if not os.path.exists(directory):
        return []
    files = []
    for ext in IMG_EXTENSIONS:
        files += glob.glob(os.path.join(directory, ext))
    return sorted(files)


def main():
    K, dist = load_camera_params()

    image_files = get_image_files(INPUT_IMG_DIR)

    if image_files:
        print(f'  input_images/ 에서 {len(image_files)}장 발견 → 이미지 모드')
        run_image_mode(K, dist, image_files)
    else:
        print('  input_images/ 없음 → 웹캠 모드')
        run_webcam_mode(K, dist)


if __name__ == '__main__':
    main()