import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 Composition Analysis Engine")

# ---------------------------
# 분석 함수들
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
        round(horizontal / total, 2),
        round(vertical / total, 2),
        round(diagonal / total, 2),
    )

def analyze_visual_weight(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    ys, xs = np.nonzero(edges)

    if len(xs) == 0:
        return w//2, h//2

    return int(np.mean(xs)), int(np.mean(ys))

def analyze_symmetry(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    flipped = cv2.flip(gray, 1)
    diff = np.mean(np.abs(gray - flipped))
    return round(1 - (diff / 255), 2)

def analyze_vertical_balance(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h = edges.shape[0]
    top = np.sum(edges[:h//2, :])
    bottom = np.sum(edges[h//2:, :])
    total = top + bottom

    if total == 0:
        return 0.5, 0.5

    return round(top/total,2), round(bottom/total,2)

# ---------------------------
# 해석 함수
# ---------------------------

def interpret_composition(h_ratio, v_ratio, d_ratio, cx, cy, w, h, symmetry, top_ratio, bottom_ratio):
    interpretations = []

    # 선 방향 해석
    if d_ratio > 0.45:
        interpretations.append("대각선 구도 성향이 강합니다.")
    elif h_ratio > 0.45:
        interpretations.append("수평 안정 구도입니다.")
    elif v_ratio > 0.45:
        interpretations.append("수직 구조 구도입니다.")
    else:
        interpretations.append("혼합 구도 성향입니다.")

    # 무게 중심 해석
    if cx < w*0.4:
        interpretations.append("시각적 무게가 좌측에 치우쳐 있습니다.")
    elif cx > w*0.6:
        interpretations.append("시각적 무게가 우측에 치우쳐 있습니다.")
    else:
        interpretations.append("시각적 무게가 중앙에 가깝습니다.")

    if cy < h*0.4:
        interpretations.append("상단 강조 구도입니다.")
    elif cy > h*0.6:
        interpretations.append("하단 강조 구도입니다.")

    # 대칭성
    if symmetry > 0.7:
        interpretations.append("대칭 구도 성향이 강합니다.")
    elif symmetry < 0.4:
        interpretations.append("비대칭 구도입니다.")

    # 상하 여백
    if top_ratio > 0.6:
        interpretations.append("상단 요소가 많이 강조되어 있습니다.")
    elif bottom_ratio > 0.6:
        interpretations.append("하단 요소가 많이 강조되어 있습니다.")

    return interpretations

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

    h_ratio, v_ratio, d_ratio = analyze_line_directions(image_np)
    cx, cy = analyze_visual_weight(image_np)
    symmetry = analyze_symmetry(image_np)
    top_ratio, bottom_ratio = analyze_vertical_balance(image_np)

    vis = image_np.copy()
    cv2.circle(vis, (cx, cy), 25, (255, 0, 0), 5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 원본")
        st.image(image_np, use_column_width=True)

    with col2:
        st.markdown("### 🔵 분석 표시")
        st.image(vis, use_column_width=True)

    st.markdown("---")
    st.markdown("## 🧠 구도 해석")

    interpretations = interpret_composition(
        h_ratio, v_ratio, d_ratio,
        cx, cy, w, h,
        symmetry, top_ratio, bottom_ratio
    )

    for text in interpretations:
        st.write("• " + text)