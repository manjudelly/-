import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import cv2

device = "cpu"

# ------------------
# Aesthetic Model
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

@st.cache_resource
def load_models():
    model, preprocess = clip.load("ViT-B/32", device=device)
    aesthetic_model = AestheticModel().to(device)

    try:
        state_dict = torch.load("ava+logos-l14-linearMSE.pth", map_location=device)
        aesthetic_model.load_state_dict(state_dict, strict=False)
    except:
        pass

    aesthetic_model.eval()
    return model, preprocess, aesthetic_model

model, preprocess, aesthetic_model = load_models()

# ------------------
# 점수 계산 함수
# ------------------
def aesthetic_score(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = aesthetic_model(image_features).item()
    return score

def exposure_percent(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    target = 135
    diff = target - mean
    percent = (diff / 255) * 100
    return round(percent, 1)

def saturation_percent(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mean = np.mean(sat)
    target = 100
    diff = target - mean
    percent = (diff / 255) * 100
    return round(percent, 1)

# ------------------
# 크롭 계산
# ------------------
def get_subject_center(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, _, _, maxLoc = cv2.minMaxLoc(gray)
    return maxLoc

def central_crop_box(image_np):
    h, w = image_np.shape[:2]
    cx, cy = get_subject_center(image_np)

    box_w = int(w * 0.6)
    box_h = int(h * 0.6)

    x1 = max(cx - box_w // 2, 0)
    y1 = max(cy - box_h // 2, 0)
    x2 = min(x1 + box_w, w)
    y2 = min(y1 + box_h, h)

    return x1, y1, x2, y2

def rule_of_thirds_crop_box(image_np):
    h, w = image_np.shape[:2]
    cx, cy = get_subject_center(image_np)

    target_x = w // 3
    target_y = h // 3

    shift_x = cx - target_x
    shift_y = cy - target_y

    box_w = int(w * 0.6)
    box_h = int(h * 0.6)

    x1 = max(target_x - box_w // 2, 0)
    y1 = max(target_y - box_h // 2, 0)
    x2 = min(x1 + box_w, w)
    y2 = min(y1 + box_h, h)

    return x1, y1, x2, y2

def draw_box(image_np, box):
    x1, y1, x2, y2 = box
    img_copy = image_np.copy()
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img_copy

# ------------------
# UI
# ------------------
st.title("📸 AI Photo Editor Guide")

uploaded_files = st.file_uploader(
    "이미지 업로드 (최대 8장)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    if len(uploaded_files) > 8:
        st.warning("⚠ 최대 8장까지 업로드 가능합니다.")
        st.stop()

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        aesth = aesthetic_score(image)
        bright = exposure_percent(image_np)
        sat = saturation_percent(image_np)

        st.subheader(uploaded_file.name)
        st.image(image, use_column_width=True)

        st.markdown("### 💡 추천 보정")
        st.markdown(f"- 밝기 {'+' if bright>0 else ''}{bright}%")
        st.markdown(f"- 채도 {'+' if sat>0 else ''}{sat}%")

        mode = st.radio(
            "크롭 모드 선택",
            ["중앙 안정형", "3분할 감성형"],
            key=uploaded_file.name
        )

        if mode == "중앙 안정형":
            box = central_crop_box(image_np)
        else:
            box = rule_of_thirds_crop_box(image_np)

        boxed_image = draw_box(image_np, box)
        st.image(boxed_image, use_column_width=True)

        st.divider()