import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(layout="wide")

st.title("📸 Photo Composition Editor")

# -----------------------------
# 선 방향 분석
# -----------------------------
def analyze_line_directions(image_np):

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=120,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0,0,0

    horizontal = vertical = diagonal = 0

    for line in lines:
        x1,y1,x2,y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2-y1,x2-x1)))

        if angle < 10:
            horizontal += 1
        elif 80 < angle < 100:
            vertical += 1
        else:
            diagonal += 1

    total = horizontal+vertical+diagonal

    if total == 0:
        return 0,0,0

    return horizontal/total, vertical/total, diagonal/total


# -----------------------------
# 시각적 무게 중심
# -----------------------------
def analyze_visual_weight(image_np):

    gray = cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)

    h,w = edges.shape

    ys,xs = np.nonzero(edges)

    if len(xs)==0:
        return w//2,h//2

    return int(np.mean(xs)),int(np.mean(ys))


# -----------------------------
# 크롭
# -----------------------------
def crop_around_point(image,center_x,center_y,ratio):

    h,w=image.shape[:2]

    box_w=int(w*ratio)
    box_h=int(h*ratio)

    x1=int(center_x-box_w/2)
    y1=int(center_y-box_h/2)

    x1=max(0,min(x1,w-box_w))
    y1=max(0,min(y1,h-box_h))

    return image[y1:y1+box_h,x1:x1+box_w]


# -----------------------------
# 후보 평가
# -----------------------------
def evaluate_crop(image_np,dominant):

    h,w=image_np.shape[:2]

    cx,cy=analyze_visual_weight(image_np)

    score=100

    strengths=[]
    weaknesses=[]

    center_dist=abs(cx-w/2)/w+abs(cy-h/2)/h

    if center_dist<0.15:
        strengths.append("시각적 중심이 안정적입니다.")
    else:
        score-=15
        weaknesses.append("중심이 약간 치우쳐 있습니다.")

    if dominant=="diagonal":

        _,_,d=analyze_line_directions(image_np)

        if d<0.4:
            score-=15
            weaknesses.append("대각선 구도 성향이 약해졌습니다.")
        else:
            strengths.append("대각선 구도가 유지되었습니다.")

    score=max(0,min(100,int(score)))

    return score,strengths,weaknesses


# -----------------------------
# 이미지 다운로드 변환
# -----------------------------
def image_to_bytes(img):

    pil_img=Image.fromarray(img)

    buf=io.BytesIO()

    pil_img.save(buf,format="JPEG")

    return buf.getvalue()


# -----------------------------
# 업로드
# -----------------------------
uploaded_files = st.file_uploader(
    "사진 업로드",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write("총 사진 수:", len(uploaded_files))

    # -----------------------------
    # 사진 슬라이드
    # -----------------------------
    index = st.slider(
        "사진 선택",
        0,
        len(uploaded_files)-1,
        0
    )

    file = uploaded_files[index]

    image = Image.open(file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np,use_column_width=True)

    h,w = image_np.shape[:2]

    # -----------------------------
    # 분석
    # -----------------------------
    h_ratio,v_ratio,d_ratio = analyze_line_directions(image_np)

    cx,cy = analyze_visual_weight(image_np)

    if d_ratio>0.45:
        dominant="diagonal"
    elif h_ratio>0.45:
        dominant="horizontal"
    elif v_ratio>0.45:
        dominant="vertical"
    else:
        dominant="mixed"

    st.write("감지된 구도:",dominant)

    # -----------------------------
    # 모드 선택
    # -----------------------------
    mode = st.radio(
        "구도 모드",
        ["🔥 구도 강화","⚖ 구도 안정화","🎨 구도 재구성"]
    )

    # -----------------------------
    # 후보 생성
    # -----------------------------
    if mode=="🔥 구도 강화":

        ratios=[0.85,0.75,0.65]
        targets=[(cx,cy)]*3

    elif mode=="⚖ 구도 안정화":

        ratios=[0.9,0.8,0.7]
        targets=[(w//2,h//2)]*3

    else:

        ratios=[0.75,0.7,0.65]
        targets=[
            (int(w/3),int(h/3)),
            (int(w*2/3),int(h/3)),
            (int(w/3),int(h*2/3))
        ]

    candidates=[]

    for i in range(3):

        cropped=crop_around_point(
            image_np,
            targets[i][0],
            targets[i][1],
            ratios[i]
        )

        score,strengths,weaknesses=evaluate_crop(
            cropped,
            dominant
        )

        candidates.append((cropped,score,strengths,weaknesses))

    candidates.sort(key=lambda x:x[1],reverse=True)

    # -----------------------------
    # 후보 표시
    # -----------------------------
    cols = st.columns(3)

    for i,(img,score,strengths,weaknesses) in enumerate(candidates):

        with cols[i]:

            st.image(img,use_column_width=True)

            st.write("⭐",score,"점")

            if strengths:
                for s in strengths:
                    st.write("•",s)

            if weaknesses:
                for w_ in weaknesses:
                    st.write("•",w_)

            st.download_button(
                "⬇️ 다운로드",
                data=image_to_bytes(img),
                file_name=f"edited_{i}.jpg",
                mime="image/jpeg",
                key=f"download_{index}_{i}"
            )