import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Aesthetic Model
# ------------------
class AestheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
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

@st.cache_resource
def load_models():
    model, preprocess = clip.load("ViT-L/14", device=device)
    aesthetic_model = AestheticModel().to(device)
    state_dict = torch.load("ava+logos-l14-linearMSE.pth", map_location=device)
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.eval()
    return model, preprocess, aesthetic_model

model, preprocess, aesthetic_model = load_models()

# ------------------
# 계산 함수
# ------------------
def laplacian_variance(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_saliency(image_np):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(image_np)
    return (saliency_map * 255).astype("uint8")

def sharpness_score(image_np, saliency_map):
    overall = laplacian_variance(image_np)
    _, _, _, maxLoc = cv2.minMaxLoc(saliency_map)
    x, y = maxLoc
    h, w = image_np.shape[:2]
    box = min(h, w) // 4
    subject = image_np[max(y-box,0):min(y+box,h), max(x-box,0):min(x+box,w)]
    subject_sharp = laplacian_variance(subject)

    if overall < 40:
        return -3
    elif subject_sharp < 60:
        return -1.5
    elif subject_sharp > 120 and overall > 100:
        return 1
    elif subject_sharp > 120:
        return 0.5
    else:
        return 0

def composition_score(image_np, saliency_map):
    h, w = image_np.shape[:2]
    _, _, _, maxLoc = cv2.minMaxLoc(saliency_map)
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
    std = np.std(gray)

    if mean < 60 or mean > 200:
        return -1.5
    elif 70 < mean < 180 and std > 40:
        return 0.5
    else:
        return 0

def color_score(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    mean_sat = np.mean(sat)
    std_sat = np.std(sat)
    std_hue = np.std(hue)

    mean_r = np.mean(image_np[:, :, 0])
    mean_g = np.mean(image_np[:, :, 1])
    mean_b = np.mean(image_np[:, :, 2])
    rgb_balance = np.std([mean_r, mean_g, mean_b])

    bonus = 0

    if mean_sat < 35:
        bonus -= 1.5
    elif 60 < mean_sat < 160:
        bonus += 0.7

    if std_sat > 50:
        bonus += 0.4

    if std_hue > 40:
        bonus += 0.4

    if rgb_balance > 40:
        bonus -= 0.7

    return bonus

# ------------------
# UI
# ------------------
st.title("📸 Practical Photo Selector - Ranked View")

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

            saliency_map = compute_saliency(image_np)

            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                aesthetic = aesthetic_model(image_features).item()

            sharp = sharpness_score(image_np, saliency_map)
            comp = composition_score(image_np, saliency_map)
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

    # Top 3
    st.subheader("🏆 Top 3")
    top_cols = st.columns(min(3, len(results)))
    for col, item in zip(top_cols, results[:3]):
        col.image(item["image"], use_column_width=True)
        col.markdown(f"### ⭐ {item['final']}")

    st.divider()

    # 전체 사진 + 점수 표시
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