import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 Composition Mode Engine")

# ---------------------------
# 분석 함수들
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
# 크롭 함수
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

    cx, cy = analyze_visual_weight(image_np)

    vis = image_np.copy()
    cv2.circle(vis, (cx, cy), 25, (255, 0, 0), 5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 원본")
        st.image(image_np, use_column_width=True)

    with col2:
        st.markdown("### 🔵 시각적 무게 중심")
        st.image(vis, use_column_width=True)

    st.markdown("---")
    st.markdown("## 🎛 구도 모드 선택")

    mode = st.radio(
        "모드를 선택하세요",
        ["🔥 구도 강화", "⚖ 구도 안정화", "🎨 구도 재구성"]
    )

    if mode == "🔥 구도 강화":
        target_x, target_y = cx, cy
        ratio = 0.75

    elif mode == "⚖ 구도 안정화":
        target_x, target_y = w//2, h//2
        ratio = 0.80

    else:  # 재구성
        target_x, target_y = w//3, h//3
        ratio = 0.70

    cropped = crop_around_point(image_np, target_x, target_y, ratio)

    st.markdown("## ✂ 크롭 결과")
    st.image(cropped, use_column_width=True)