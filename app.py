import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 Horizon & Crop Guide")

# ------------------
# 수평 감지 (평균 기반)
# ------------------

def detect_horizon_angle(image_np):
    h, w = image_np.shape[:2]

    scale_w = 800
    scale_h = int(h * scale_w / w)
    small = cv2.resize(image_np, (scale_w, scale_h))

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return None

    valid_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) < 15:
            valid_angles.append(angle)

    if len(valid_angles) == 0:
        return None

    return np.mean(valid_angles)

# ------------------
# 회전 적용 (부호 안정화)
# ------------------

def rotate_image(image, angle):
    if angle is None:
        return image

    h, w = image.shape[:2]
    center = (w//2, h//2)

    # 🔥 부호 반전 제거 (기존 -angle → angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        M,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )

# ------------------
# 중앙 크롭
# ------------------

def central_crop(image):
    h, w = image.shape[:2]
    box_w = int(w * 0.85)
    box_h = int(h * 0.85)

    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2

    return image[y1:y1+box_h, x1:x1+box_w]

# ------------------
# UI
# ------------------

uploaded_files = st.file_uploader(
    "이미지 업로드 (최대 8장)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if uploaded_files:

    if len(uploaded_files) > 8:
        st.warning("⚠ 최대 8장까지만 업로드 가능합니다.")
        st.stop()

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.subheader(uploaded_file.name)

        angle = detect_horizon_angle(image_np)

        if angle is not None:
            st.markdown(f"### 📐 감지된 기울기: {round(angle,2)}°")
        else:
            st.markdown("### 📐 수평선 감지 실패")

        st.markdown("### 🎛 적용 선택")

        apply_rotate = st.checkbox("📐 수평 적용", key="rot_"+uploaded_file.name)
        apply_crop = st.checkbox("✂ 중앙 크롭 적용", key="crop_"+uploaded_file.name)

        edited = image_np.copy()

        # 회전 먼저
        if apply_rotate:
            edited = rotate_image(edited, angle)

        # 크롭 나중
        if apply_crop:
            edited = central_crop(edited)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 원본")
            st.image(image_np, use_column_width=True)

        with col2:
            st.markdown("#### 미리보기")
            st.image(edited, use_column_width=True)

        st.divider()