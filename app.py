import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 AI Edit Guide (Preview Mode)")

# ------------------
# 밝기 & 채도 계산
# ------------------

def exposure_percent(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    target = 135
    diff = target - mean
    return np.clip(round((diff / 255) * 100, 1), -20, 20)

def saturation_percent(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mean = np.mean(sat)
    target = 100
    diff = target - mean
    return np.clip(round((diff / 255) * 100, 1), -20, 20)

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
# 편집 적용 함수
# ------------------

def rotate_image(image, angle):
    if angle is None:
        return image

    h, w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def apply_brightness(image, percent):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    factor = 1 + (percent/100) * 0.4  # 부드러운 보정
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_saturation(image, percent):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    factor = 1 + (percent/100) * 0.3  # 부드러운 보정
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

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

        bright = exposure_percent(image_np)
        sat = saturation_percent(image_np)
        angle = detect_horizon_angle(image_np)

        st.markdown("### 편집 제안")
        st.markdown(f"- 밝기 {'+' if bright>0 else ''}{bright}%")
        st.markdown(f"- 채도 {'+' if sat>0 else ''}{sat}%")

        if angle is not None:
            st.markdown(f"- 수평 보정: {round(angle,2)}°")

        st.markdown("### 단계별 적용 선택")

        apply_rotate = st.checkbox("수평 적용", key="rot_"+uploaded_file.name)
        apply_crop = st.checkbox("크롭 적용 (중앙)", key="crop_"+uploaded_file.name)
        apply_bright = st.checkbox("밝기 적용", key="bright_"+uploaded_file.name)
        apply_sat = st.checkbox("채도 적용", key="sat_"+uploaded_file.name)

        edited = image_np.copy()

        # 순서 중요
        if apply_rotate:
            edited = rotate_image(edited, angle)

        if apply_crop:
            edited = central_crop(edited)

        if apply_bright:
            edited = apply_brightness(edited, bright)

        if apply_sat:
            edited = apply_saturation(edited, sat)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 원본")
            st.image(image_np, use_column_width=True)

        with col2:
            st.markdown("#### 미리보기")
            st.image(edited, use_column_width=True)

        st.divider()