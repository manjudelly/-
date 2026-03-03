import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("📸 Composition Mode Engine v3")

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

    return horizontal/total, vertical/total, diagonal/total


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
# 4️⃣ 후보 평가 함수
# ---------------------------
def evaluate_crop(image_np, dominant_type):

    h, w = image_np.shape[:2]
    cx, cy = analyze_visual_weight(image_np)
    h_ratio, v_ratio, d_ratio = analyze_line_directions(image_np)

    score = 100
    strengths = []
    weaknesses = []

    # 1️⃣ 중심 안정성 점수
    center_dist = abs(cx - w/2) / w + abs(cy - h/2) / h
    center_penalty = center_dist * 40
    score -= center_penalty

    if center_dist < 0.15:
        strengths.append("시각적 중심이 안정적으로 배치되었습니다.")
    else:
        weaknesses.append("시각적 중심이 프레임 중앙에서 다소 벗어나 있습니다.")

    # 2️⃣ 구도 유지 점수
    if dominant_type == "diagonal":
        if d_ratio > 0.4:
            strengths.append("대각선 구도 성향이 잘 유지되었습니다.")
        else:
            score -= 15
            weaknesses.append("대각선 구도 성향이 약화되었습니다.")

    if dominant_type == "horizontal":
        if h_ratio > 0.4:
            strengths.append("수평 안정 구도가 유지되었습니다.")
        else:
            score -= 15
            weaknesses.append("수평 구조가 다소 약해졌습니다.")

    # 3️⃣ 여백 균형 점수
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    top = np.sum(edges[:h//2, :])
    bottom = np.sum(edges[h//2:, :])
    balance = abs(top - bottom) / (top + bottom + 1)

    if balance < 0.2:
        strengths.append("상하 균형이 비교적 좋습니다.")
    else:
        score -= 10
        weaknesses.append("상하 균형이 다소 치우쳐 있습니다.")

    score = max(0, min(100, int(score)))

    return score, strengths, weaknesses


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

    # 구도 타입 판별
    if d_ratio > 0.45:
        dominant = "diagonal"
    elif h_ratio > 0.45:
        dominant = "horizontal"
    elif v_ratio > 0.45:
        dominant = "vertical"
    else:
        dominant = "mixed"

    st.markdown(f"### 🧠 감지된 구도 타입: {dominant}")

    mode = st.radio(
        "모드 선택",
        ["🔥 구도 강화", "⚖ 구도 안정화", "🎨 구도 재구성"]
    )

    candidates = []

    if mode == "🔥 구도 강화":
        ratios = [0.85, 0.75, 0.65]
        targets = [(cx, cy)] * 3

    elif mode == "⚖ 구도 안정화":
        ratios = [0.90, 0.80, 0.70]
        targets = [(w//2, h//2)] * 3

    else:  # 재구성
        ratios = [0.75, 0.70, 0.65]
        targets = [
            (int(w*1/3), int(h*1/3)),
            (int(w*2/3), int(h*1/3)),
            (int(w*1/3), int(h*2/3))
        ]

    for i in range(3):
        cropped = crop_around_point(image_np, targets[i][0], targets[i][1], ratios[i])
        score, strengths, weaknesses = evaluate_crop(cropped, dominant)
        candidates.append((cropped, score, strengths, weaknesses))

    st.markdown("## ✂ 후보 비교")

    cols = st.columns(3)

    for idx, (img, score, strengths, weaknesses) in enumerate(candidates):
        with cols[idx]:
            st.image(img, use_column_width=True)
            st.markdown(f"### ⭐ {score}점")

            if strengths:
                st.markdown("**강점**")
                for s in strengths:
                    st.write("• " + s)

            if weaknesses:
                st.markdown("**보완점**")
                for w_ in weaknesses:
                    st.write("• " + w_)