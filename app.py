import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 Composition Mode Engine v2")

# ---------------------------
# 1️⃣ 선 방향 분석
# ---------------------------
def analyze_line_directions(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=120,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0, 0, 0

    horizontal = vertical = diagonal = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angle < 10:
            horizontal += 1
        elif 80 < angle < 100:
            vertical += 1
        else:
            diagonal += 1

    total = horizontal + vertical + diagonal
    if total == 0:
        return 0, 0, 0

    return (
        horizontal / total,
        vertical / total,
        diagonal / total
    )

# ---------------------------
# 2️⃣ 시각적 무게 중심
# ---------------------------
def analyze_visual_weight(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    ys, xs = np.nonzero(edges)

    if len(xs) == 0:
        return w//2, h//2

    return int(np.mean(xs)), int(np.mean(ys))

# ---------------------------
# 3️⃣ 크롭 함수
# ---------------------------
def crop_around_point(image, center_x, center_y, ratio):
    h, w = image.shape[:2]

    box_w = int(w * ratio)
    box_h = int(h * ratio)

    x1 = int(center_x - box_w/2)
    y1 = int(center_y - box_h/2)

    x1 = max(0, min(x1, w - box_w))
    y1 = max(0, min(y1, h - box_h))

    return image[y1:y1+box_h, x1:x1+box_w]

# ---------------------------
# UI
# ---------------------------
uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # 분석
    h_ratio, v_ratio, d_ratio = analyze_line_directions(image_np)
    cx, cy = analyze_visual_weight(image_np)

    # 구도 타입 판별
    if d_ratio > 0.45:
        dominant = "diagonal"
    elif h_ratio > 0.45:
        dominant = "horizontal"
    elif v_ratio > 0.45:
        dominant = "vertical"
    else:
        dominant = "mixed"

    # 시각화
    vis = image_np.copy()
    cv2.circle(vis, (cx, cy), 25, (255, 0, 0), 5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 원본")
        st.image(image_np, use_column_width=True)

    with col2:
        st.markdown("### 🔵 무게 중심")
        st.image(vis, use_column_width=True)

    st.markdown("---")
    st.markdown("## 🧠 분석 결과")

    st.write(f"수평 비율: {round(h_ratio,2)}")
    st.write(f"수직 비율: {round(v_ratio,2)}")
    st.write(f"대각선 비율: {round(d_ratio,2)}")
    st.write(f"구도 타입: {dominant}")

    st.markdown("---")
    st.markdown("## 🎛 구도 모드 선택")

    mode = st.radio(
        "모드 선택",
        ["🔥 구도 강화", "⚖ 구도 안정화", "🎨 구도 재구성"]
    )

    # ---------------------------
    # 모드별 전략 (구도 타입 반영)
    # ---------------------------

    if mode == "🔥 구도 강화":

        if dominant == "diagonal":
            ratio = 0.75
            target_x, target_y = cx, cy

        elif dominant == "horizontal":
            ratio = 0.80
            target_x, target_y = w//2, cy  # 지평선 유지

        elif dominant == "vertical":
            ratio = 0.80
            target_x, target_y = cx, h//2  # 수직 강조

        else:
            ratio = 0.75
            target_x, target_y = cx, cy

    elif mode == "⚖ 구도 안정화":

        ratio = 0.85
        target_x, target_y = w//2, h//2  # 완전 중앙

    else:  # 🎨 재구성

        ratio = 0.70

        if dominant == "diagonal":
            target_x, target_y = int(w*2/3), int(h*1/3)
        elif dominant == "horizontal":
            target_x, target_y = w//2, int(h*2/3)
        else:
            target_x, target_y = int(w*1/3), int(h*1/3)

    cropped = crop_around_point(image_np, target_x, target_y, ratio)

    st.markdown("## ✂ 크롭 결과")
    st.image(cropped, use_column_width=True)