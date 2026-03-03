import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 Composition Analysis Engine")

# ---------------------------
# 1️⃣ 선 방향 분석
# ---------------------------
def analyze_line_directions(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0, 0, 0

    horizontal = 0
    vertical = 0
    diagonal = 0

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
        round(horizontal / total, 2),
        round(vertical / total, 2),
        round(diagonal / total, 2),
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
        return w // 2, h // 2

    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))

    return center_x, center_y

# ---------------------------
# 3️⃣ 좌우 대칭 점수
# ---------------------------
def analyze_symmetry(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    flipped = cv2.flip(gray, 1)

    diff = np.mean(np.abs(gray - flipped))
    symmetry_score = 1 - (diff / 255)

    return round(symmetry_score, 2)

# ---------------------------
# 4️⃣ 상/하 에지 밀도
# ---------------------------
def analyze_vertical_balance(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h = edges.shape[0]
    top_density = np.sum(edges[:h//2, :])
    bottom_density = np.sum(edges[h//2:, :])

    total = top_density + bottom_density
    if total == 0:
        return 0.5, 0.5

    return (
        round(top_density / total, 2),
        round(bottom_density / total, 2),
    )

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

    # 분석 수행
    h_ratio, v_ratio, d_ratio = analyze_line_directions(image_np)
    cx, cy = analyze_visual_weight(image_np)
    symmetry = analyze_symmetry(image_np)
    top_ratio, bottom_ratio = analyze_vertical_balance(image_np)

    # 무게 중심 표시 이미지
    vis = image_np.copy()
    cv2.circle(vis, (cx, cy), 25, (255, 0, 0), 5)

    # 🔥 좌우 비교 레이아웃
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 원본")
        st.image(image_np, use_column_width=True)

    with col2:
        st.markdown("### 🔵 분석 표시")
        st.image(vis, use_column_width=True)

    st.markdown("---")
    st.markdown("## 📊 분석 결과")

    st.write(f"수평 선 비율: {h_ratio}")
    st.write(f"수직 선 비율: {v_ratio}")
    st.write(f"대각선 선 비율: {d_ratio}")

    st.write(f"시각적 무게 중심: ({cx}, {cy})")

    st.write(f"좌우 대칭 점수 (1에 가까울수록 대칭): {symmetry}")

    st.write(f"상단 에지 비율: {top_ratio}")
    st.write(f"하단 에지 비율: {bottom_ratio}")