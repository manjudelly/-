import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 AI Edit Guide")

# ------------------
# 밝기 & 채도 계산
# ------------------

def exposure_percent(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    target = 135
    diff = target - mean
    return round((diff / 255) * 100, 1)

def saturation_percent(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mean = np.mean(sat)
    target = 100
    diff = target - mean
    return round((diff / 255) * 100, 1)

# ------------------
# 크롭 계산
# ------------------

def get_subject_center(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, _, _, maxLoc = cv2.minMaxLoc(gray)
    return maxLoc

def central_crop_box(image_np):
    h, w = image_np.shape[:2]
    cx, cy = get_subject_center(image_np)
    box_w = int(w * 0.6)
    box_h = int(h * 0.6)
    x1 = max(cx - box_w // 2, 0)
    y1 = max(cy - box_h // 2, 0)
    x2 = min(x1 + box_w, w)
    y2 = min(y1 + box_h, h)
    return x1, y1, x2, y2

def rule_of_thirds_crop_box(image_np):
    h, w = image_np.shape[:2]
    target_x = w // 3
    target_y = h // 3
    box_w = int(w * 0.6)
    box_h = int(h * 0.6)
    x1 = max(target_x - box_w // 2, 0)
    y1 = max(target_y - box_h // 2, 0)
    x2 = min(x1 + box_w, w)
    y2 = min(y1 + box_h, h)
    return x1, y1, x2, y2

# ------------------
# 수평 감지 (평균 기반)
# ------------------

def detect_horizon_line(image_np):
    h, w = image_np.shape[:2]

    # 계산용으로 축소
    scale_w = 800
    scale_h = int(h * scale_w / w)
    small = cv2.resize(image_np, (scale_w, scale_h))

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return None, None, small

    valid_angles = []
    valid_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # 거의 수평인 선만 사용
        if abs(angle) < 15:
            valid_angles.append(angle)
            valid_lines.append((x1, y1, x2, y2))

    if len(valid_angles) == 0:
        return None, None, small

    mean_angle = np.mean(valid_angles)

    # 평균 각도에 가장 가까운 선 선택
    closest_idx = np.argmin(np.abs(np.array(valid_angles) - mean_angle))
    best_line = valid_lines[closest_idx]

    return best_line, mean_angle, small

# ------------------
# UI
# ------------------

uploaded_files = st.file_uploader(
    "이미지 업로드 (최대 8장)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    if len(uploaded_files) > 8:
        st.warning("⚠ 최대 8장까지만 업로드 가능합니다.")
        st.stop()

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file)
        image_np = np.array(image)

        st.subheader(uploaded_file.name)

        # 밝기 / 채도 제안
        bright = exposure_percent(image_np)
        sat = saturation_percent(image_np)

        st.markdown("### 💡 편집 제안")
        st.markdown(f"- 밝기 {'+' if bright > 0 else ''}{bright}%")
        st.markdown(f"- 채도 {'+' if sat > 0 else ''}{sat}%")

        # 상태 키
        crop_key = "crop_" + uploaded_file.name
        horizon_key = "horizon_" + uploaded_file.name

        if crop_key not in st.session_state:
            st.session_state[crop_key] = False
        if horizon_key not in st.session_state:
            st.session_state[horizon_key] = False

        # 버튼
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✂ 크롭 토글", key="btn_crop_"+uploaded_file.name):
                st.session_state[crop_key] = not st.session_state[crop_key]

        with col2:
            if st.button("📐 수평 토글", key="btn_horizon_"+uploaded_file.name):
                st.session_state[horizon_key] = not st.session_state[horizon_key]

        # ---- 통합 표시 이미지 생성 ----
        display_image = image_np.copy()

        # 크롭 적용
        if st.session_state[crop_key]:

            mode = st.radio(
                "크롭 모드",
                ["중앙 안정형", "3분할 감성형"],
                key="radio_"+uploaded_file.name
            )

            if mode == "중앙 안정형":
                box = central_crop_box(display_image)
            else:
                box = rule_of_thirds_crop_box(display_image)

            x1, y1, x2, y2 = box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # 수평 적용
        if st.session_state[horizon_key]:

            line, angle, small = detect_horizon_line(image_np)

            if line is not None:
                x1, y1, x2, y2 = line

                # 원본 크기에 맞게 좌표 비율 보정
                h, w = image_np.shape[:2]
                scale_w = 800
                scale_h = int(h * scale_w / w)

                ratio_x = w / scale_w
                ratio_y = h / scale_h

                x1 = int(x1 * ratio_x)
                x2 = int(x2 * ratio_x)
                y1 = int(y1 * ratio_y)
                y2 = int(y2 * ratio_y)

                cv2.line(display_image, (x1, y1), (x2, y2), (0, 0, 255), 4)

                if abs(angle) >= 0.1:
                    direction = "시계 방향" if angle < 0 else "반시계 방향"
                    st.warning(f"{direction}으로 약 {abs(round(angle,2))}° 회전 추천")
                else:
                    st.success("거의 완벽한 수평입니다 (±0.1° 이내)")
            else:
                st.info("수평 기준선을 찾기 어렵습니다.")

        st.image(display_image, use_column_width=True)
        st.divider()