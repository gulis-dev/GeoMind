import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import pandas as pd
import os


class DirectorModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.fc = torch.nn.Linear(10, 13)

    def forward(self, x): return self.fc(torch.randn(x.size(0), 10))


class ExpertModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.fc = torch.nn.Linear(10, 2)

    def forward(self, x): return self.fc(torch.randn(x.size(0), 10))


REGION_BOUNDING_BOXES = {
    0: {"lat_min": 40.0, "lat_max": 50.0, "lon_min": 10.0, "lon_max": 20.0},
    1: {"lat_min": 45.0, "lat_max": 55.0, "lon_min": 20.0, "lon_max": 30.0},
    2: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    3: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    4: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    5: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    6: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    7: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    8: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    9: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    10: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    11: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    12: {"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0}
}

preprocess_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def denormalize_coords(norm_lat, norm_lon, region_id):
    if region_id not in REGION_BOUNDING_BOXES:
        raise ValueError("Invalid region ID")

    bbox = REGION_BOUNDING_BOXES[region_id]

    lat = norm_lat * (bbox['lat_max'] - bbox['lat_min']) + bbox['lat_min']
    lon = norm_lon * (bbox['lon_max'] - bbox['lon_min']) + bbox['lon_min']
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

    director_model = DirectorModel()
    # director_model.load_state_dict(torch.load(director_path, map_location="cpu"))
    # director_model.eval()

    expert_models = {}
    for i in range(13):
        expert_path = os.path.join(experts_dir, f"expert_regressor_{i}.pth")
        if not os.path.exists(expert_path):
            st.warning(f"Missing expert model for region {i} at {expert_path}")
            continue

        expert = ExpertModel()
        # expert.load_state_dict(torch.load(expert_path, map_location="cpu"))
        # expert.eval()
        expert_models[i] = expert

    return director_model, expert_models


director_model, expert_models = load_models()


def run_inference(image):
    tensor = preprocess_transform(image).unsqueeze(0)

    with torch.no_grad():
        director_logits = director_model(tensor)
        region_id = torch.argmax(director_logits, dim=1).item()

    if region_id not in expert_models:
        st.error(f"Region {region_id} detected, but missing expert model file.")
        return None, None

    expert_model = expert_models[region_id]

    with torch.no_grad():
        norm_coords = expert_model(tensor)
        norm_lat = norm_coords[0][0].item()
        norm_lon = norm_coords[0][1].item()

    lat, lon = denormalize_coords(norm_lat, norm_lon, region_id)

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

                    gmaps_link = f"https://www.google.com/maps?q={lat},{lon}"
                    st.link_button("Open in Google Maps", gmaps_link)

st.sidebar.header("About This Project")
st.sidebar.write("""
This application demonstrates a two-stage pipeline (Director + Experts) for image geolocation.
""")
st.sidebar.write("**Authors:** Oskar Andrukiewicz, Piotr Kaptur")
st.sidebar.link_button("GitHub Repository", "https://github.com/gulis-dev/GeoMind")