import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import cv2

device = "cpu"

# ------------------
# Aesthetic Model (512차원용)
# ------------------
class AestheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# ------------------
# 모델 로딩
# ------------------
@st.cache_resource
def load_models():
    model, preprocess = clip.load("ViT-B/32", device=device)

    aesthetic_model = AestheticModel().to(device)

    # weight 로드 (shape 안 맞으면 에러 날 수 있음)
    try:
        state_dict = torch.load(
            "ava+logos-l14-linearMSE.pth",
            map_location=device
        )
        aesthetic_model.load_state_dict(state_dict, strict=False)
    except:
        pass

    aesthetic_model.eval()

    return model, preprocess, aesthetic_model

model, preprocess, aesthetic_model = load_models()

# ------------------
# 분석 함수들
# ------------------
def laplacian_variance(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def sharpness_score(image_np):
    val = laplacian_variance(image_np)

    if val < 40:
        return -2
    elif val > 150:
        return 1
    else:
        return 0

def composition_score(image_np):
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, _, _, maxLoc = cv2.minMaxLoc(gray)
    x, y = maxLoc

    col = x // (w // 3)
    row = y // (h // 3)

    if row == 1 and col == 1:
        return 0.5
    elif (row in [0,2] and col in [0,2]):
        return 1.0
    else:
        return 0

def exposure_score(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)

    if mean < 60 or mean > 200:
        return -1
    elif 80 < mean < 170:
        return 0.5
    else:
        return 0

def color_score(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mean_sat = np.mean(sat)

    if mean_sat < 35:
        return -1
    elif 60 < mean_sat < 150:
        return 0.7
    else:
        return 0

def aesthetic_score(image):
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = aesthetic_model(image_features).item()

    return score

# ------------------
# UI
# ------------------
st.title("📸 Photo Selector - Cloud Version")

uploaded_files = st.file_uploader(
    "이미지 업로드",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []

    with st.spinner("🔍 분석 중..."):
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            aesthetic = aesthetic_score(image)
            sharp = sharpness_score(image_np)
            comp = composition_score(image_np)
            expo = exposure_score(image_np)
            color = color_score(image_np)

            final = (aesthetic * 0.3) + sharp + comp + expo + color

            results.append({
                "image": image,
                "name": uploaded_file.name,
                "final": round(final, 2),
                "aesthetic": round(aesthetic, 2),
                "sharp": sharp,
                "comp": comp,
                "expo": expo,
                "color": color
            })

    results = sorted(results, key=lambda x: x["final"], reverse=True)

    st.subheader("🏆 Top 3")
    cols = st.columns(min(3, len(results)))

    for col, item in zip(cols, results[:3]):
        col.image(item["image"], use_column_width=True)
        col.markdown(f"### ⭐ {item['final']}")

    st.divider()

    st.subheader("📊 전체 결과")

    for item in results:
        st.image(item["image"], use_column_width=True)
        st.markdown(f"""
        **⭐ 최종 점수:** {item['final']}  
        - Aesthetic: {item['aesthetic']}  
        - Sharpness: {item['sharp']}  
        - Composition: {item['comp']}  
        - Exposure: {item['expo']}  
        - Color: {item['color']}
        """)
        st.divider()