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

def draw_box(image_np, box):
    img_copy = image_np.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img_copy

# ------------------
# 수평 감지
# ------------------

def detect_horizon_line(image_np):
    h, w = image_np.shape[:2]
    small = cv2.resize(image_np, (800, int(h * 800 / w)))
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

    best_line = None
    smallest_angle = 90

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < abs(smallest_angle):
            smallest_angle = angle
            best_line = (x1, y1, x2, y2)

    return best_line, smallest_angle, small

def draw_horizon_line(image_np):
    line, angle, small = detect_horizon_line(image_np)
    if line is None:
        return None, None
    x1, y1, x2, y2 = line
    cv2.line(small, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return small, angle

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
        st.image(image, use_column_width=True)

        bright = exposure_percent(image_np)
        sat = saturation_percent(image_np)

        st.markdown("### 편집 제안")
        st.markdown(f"- 밝기 {'+' if bright > 0 else ''}{bright}%")
        st.markdown(f"- 채도 {'+' if sat > 0 else ''}{sat}%")

        # ------------------
        # 크롭 토글
        # ------------------

        crop_key = "crop_" + uploaded_file.name
        if crop_key not in st.session_state:
            st.session_state[crop_key] = False

        if st.button("크롭 가이드", key="btn_crop_"+uploaded_file.name):
            st.session_state[crop_key] = not st.session_state[crop_key]

        if st.session_state[crop_key]:

            mode = st.radio(
                "크롭 모드",
                ["중앙 안정형", "3분할 감성형"],
                key="radio_"+uploaded_file.name
            )

            if mode == "중앙 안정형":
                box = central_crop_box(image_np)
            else:
                box = rule_of_thirds_crop_box(image_np)

            boxed_image = draw_box(image_np, box)
            st.image(boxed_image, use_column_width=True)

        # ------------------
        # 수평 토글
        # ------------------

        horizon_key = "horizon_" + uploaded_file.name
        if horizon_key not in st.session_state:
            st.session_state[horizon_key] = False

        if st.button("수평 가이드", key="btn_horizon_"+uploaded_file.name):
            st.session_state[horizon_key] = not st.session_state[horizon_key]

        if st.session_state[horizon_key]:
            horizon_img, angle = draw_horizon_line(image_np)

            if horizon_img is None:
                st.info("수평 기준선을 찾기 어렵습니다.")
            else:
                st.image(horizon_img, use_column_width=True)

                if abs(angle) < 0.5:
                    st.success("수평이 잘 맞습니다 (±0.5° 이내)")
                else:
                    direction = "시계 방향" if angle < 0 else "반시계 방향"
                    st.warning(f"{direction}으로 약 {abs(round(angle,1))}° 회전 추천")

        st.divider()