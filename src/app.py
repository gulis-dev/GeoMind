import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import pandas as pd
import os
from efficientnet_pytorch import EfficientNet
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

REGION_PARAMS = {
    0: {'lat_min': 15.0, 'lat_max': 70.0, 'lon_min': -168.0, 'lon_max': -52.0},
    1: {'lat_min': -56.0, 'lat_max': 33.0, 'lon_min': -118.0, 'lon_max': -34.0},
    2: {'lat_min': 36.0, 'lat_max': 71.0, 'lon_min': -25.0, 'lon_max': 40.0},
    3: {'lat_min': 34.0, 'lat_max': 47.0, 'lon_min': -10.0, 'lon_max': 28.0},
    4: {'lat_min': 41.0, 'lat_max': 60.0, 'lon_min': 12.0, 'lon_max': 41.0},
    5: {'lat_min': 41.0, 'lat_max': 82.0, 'lon_min': 19.0, 'lon_max': 180.0},
    6: {'lat_min': 20.0, 'lat_max': 46.0, 'lon_min': 122.0, 'lon_max': 153.0},
    7: {'lat_min': -11.0, 'lat_max': 23.0, 'lon_min': 95.0, 'lon_max': 155.0},
    8: {'lat_min': 5.0, 'lat_max': 37.0, 'lon_min': 60.0, 'lon_max': 97.0},
    9: {'lat_min': -35.0, 'lat_max': 37.0, 'lon_min': -18.0, 'lon_max': 52.0},
    10: {'lat_min': 12.0, 'lat_max': 34.0, 'lon_min': 33.0, 'lon_max': 60.0},
    11: {'lat_min': -47.0, 'lat_max': -9.0, 'lon_min': 112.0, 'lon_max': 180.0},
    12: {'lat_min': -50.0, 'lat_max': 80.0, 'lon_min': -180.0, 'lon_max': 180.0},
}


def crop_center_640x480(image: Image.Image) -> Image.Image:
    w, h = image.size
    cw, ch = 640, 480
    if w < cw or h < ch:
        st.warning(f"Image is smaller than 640x480 ({w}x{h}). Resizing...")
        image = image.resize((cw, ch), Image.LANCZOS)

    w, h = image.size
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch
    return image.crop((left, top, right, bottom))


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = crop_center_640x480(image)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def load_director(director_path, device=DEVICE):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, len(REGION_PARAMS))
    model.load_state_dict(torch.load(director_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_expert(region_id, experts_dir, device=DEVICE):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Sequential(
        torch.nn.Linear(model._fc.in_features, 2),
        torch.nn.Sigmoid()
    )
    weights_path = f'{experts_dir}/expert_regressor_{region_id}.pth'
    if not os.path.exists(weights_path):
        st.error(f"Missing expert model file at: {weights_path}")
        return None

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def denormalize(pred, region_id):
    region_params = REGION_PARAMS[region_id]
    lat = pred[0] * (region_params['lat_max'] - region_params['lat_min']) + region_params['lat_min']
    lon = pred[1] * (region_params['lon_max'] - region_params['lon_min']) + region_params['lon_min']
    return lat, lon


@st.cache_resource
def load_models():
    director_path = "saved_models/director/efficientnet_b0_director_v3.pth"
    experts_dir = "saved_models/experts/v2"

    if not os.path.exists(director_path):
        st.error(f"Director model not found at {director_path}")
        return None, None
    if not os.path.exists(experts_dir):
        st.error(f"Experts directory not found at {experts_dir}")
        return None, None

    director_model = load_director(director_path, DEVICE)

    expert_models = {}
    for i in range(len(REGION_PARAMS)):
        expert = load_expert(i, experts_dir, DEVICE)
        if expert is not None:
            expert_models[i] = expert

    return director_model, expert_models


director_model, expert_models = load_models()


def run_inference(image: Image.Image):
    tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        director_logits = director_model(tensor)
        region_id = director_logits.argmax(dim=1).item()

    if region_id not in expert_models:
        st.error(f"Region {region_id} detected, but missing expert model file.")
        return None, None

    expert_model = expert_models[region_id]

    with torch.no_grad():
        pred_exp = expert_model(tensor).cpu().numpy()[0]

    lat, lon = denormalize(pred_exp, region_id)

    return region_id, (lat, lon)


st.set_page_config(page_title="GeoMind AI")
st.title("GeoMind – GeoGuessr AI")
st.write("Upload an image to predict its approximate geographic location.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Localize Image"):
        if not director_model or not expert_models:
            st.error("Models are not loaded correctly. Check logs.")
        else:
            with st.spinner("Analyzing... (Director → Expert)"):
                region_id, coords = run_inference(image)

                if region_id is not None and coords is not None:
                    lat, lon = coords
                    st.subheader(f"Predicted Region: {region_id}")
                    st.subheader(f"Coordinates: ({lat:.4f}, {lon:.4f})")

                    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                    st.map(map_data, zoom=4)

                    gmaps_link = f"https.www.google.com/maps?q={lat},{lon}"
                    st.link_button("Open in Google Maps", gmaps_link)

st.sidebar.header("About This Project")
st.sidebar.write("""
This application demonstrates a two-stage pipeline (Director + Experts) for image geolocation.
""")
st.sidebar.write("**Authors:** Oskar Andrukiewicz, Piotr Kaptur")
st.sidebar.link_button("GitHub Repository", "https://github.com/gulis-dev/GeoMind")