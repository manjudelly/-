import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(layout="wide")
st.title("📸 Smart Composition Editor")

# -------------------------
# edge 기반 중심
# -------------------------
def get_visual_center(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)

    ys,xs = np.nonzero(edges)

    h,w = edges.shape

    if len(xs)==0:
        return w//2,h//2

    return int(np.mean(xs)),int(np.mean(ys))


# -------------------------
# 크롭
# -------------------------
def crop(img,cx,cy,ratio):

    h,w = img.shape[:2]

    cw = int(w*ratio)
    ch = int(h*ratio)

    x1 = int(cx-cw/2)
    y1 = int(cy-ch/2)

    x1=max(0,min(x1,w-cw))
    y1=max(0,min(y1,h-ch))

    return img[y1:y1+ch,x1:x1+cw]


# -------------------------
# 다양한 크롭 후보 생성
# -------------------------
def generate_candidates(img):

    h,w = img.shape[:2]

    cx,cy = get_visual_center(img)

    candidates=[]

    # Rule of thirds
    candidates.append(crop(img,int(w/3),int(h/3),0.7))
    candidates.append(crop(img,int(2*w/3),int(2*h/3),0.7))

    # Golden ratio
    candidates.append(crop(img,int(w*0.618),int(h*0.618),0.75))

    # Diagonal
    candidates.append(crop(img,int(w*0.75),int(h*0.25),0.7))

    # Tight focus
    candidates.append(crop(img,cx,cy,0.6))

    return candidates


# -------------------------
# 점수 평가
# -------------------------
def evaluate(img):

    h,w = img.shape[:2]

    cx,cy = get_visual_center(img)

    score=100

    # 중심
    dist = abs(cx-w/2)/w + abs(cy-h/2)/h
    score -= dist*30

    # 좌우 균형
    left = np.sum(img[:,:w//2])
    right = np.sum(img[:,w//2:])
    score -= abs(left-right)/(left+right+1)*20

    # 상하 균형
    top = np.sum(img[:h//2,:])
    bottom = np.sum(img[h//2:,:])
    score -= abs(top-bottom)/(top+bottom+1)*15

    return int(max(0,min(100,score)))


# -------------------------
# 다운로드 변환
# -------------------------
def img_bytes(img):

    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf,format="JPEG")

    return buf.getvalue()


# -------------------------
# 업로드
# -------------------------
files = st.file_uploader(
    "사진 업로드",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if files:

    index = st.slider(
        "사진 선택",
        0,
        len(files)-1,
        0
    )

    img = Image.open(files[index]).convert("RGB")
    img = np.array(img)

    st.image(img,use_column_width=True)

    candidates = generate_candidates(img)

    scored=[]

    for c in candidates:
        s = evaluate(c)
        scored.append((c,s))

    # 점수 정렬
    scored.sort(key=lambda x:x[1],reverse=True)

    # 상위 3개
    scored = scored[:3]

    tabs = st.tabs(["후보1","후보2","후보3"])

    for i,tab in enumerate(tabs):

        with tab:

            im,sc = scored[i]

            st.image(im,use_column_width=True)

            st.write("⭐",sc,"점")

            st.download_button(
                "⬇️ 다운로드",
                data=img_bytes(im),
                file_name=f"crop_{i}.jpg",
                mime="image/jpeg",
                key=f"d{i}"
            )