import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from efficientnet_pytorch import EfficientNet
import math

MODEL_DIR = '../../saved_models/experts/v2'
DIRECTOR_PATH = '../../saved_models/director/efficientnet_b0_director_v2.pth'
INPUT_DIR = 'images'
CSV_PATH = 'metadata.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

REGION_MAP = {
    0: "North America",
    1: "Latin America",
    2: "Western & Northern Europe",
    3: "Southern Europe",
    4: "Eastern Europe & Balkans",
    5: "Russia & Cyrillic",
    6: "East Asia",
    7: "Southeast Asia",
    8: "South Asia",
    9: "Africa",
    10: "Arabia",
    11: "Oceania",
    12: "Rare Regions",
}

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
        raise ValueError(f"Obraz musi mieć co najmniej 640x480 px (jest {w}x{h})")
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch
    return image.crop((left, top, right, bottom))


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = crop_center_640x480(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def load_director(device=DEVICE):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, len(REGION_MAP))
    model.load_state_dict(torch.load(DIRECTOR_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_expert(region_id, device=DEVICE):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Sequential(
        torch.nn.Linear(model._fc.in_features, 2),
        torch.nn.Sigmoid()
    )
    weights_path = f'{MODEL_DIR}/expert_regressor_{region_id}.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def denormalize(pred, region_params):
    lat = pred[0] * (region_params['lat_max'] - region_params['lat_min']) + region_params['lat_min']
    lon = pred[1] * (region_params['lon_max'] - region_params['lon_min']) + region_params['lon_min']
    return lat, lon


def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


if __name__ == "__main__":
    try:
        ground_truth = pd.read_csv(CSV_PATH)
        ground_truth_dict = {
            row['filename']: {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'superRegion': row['superRegion'],
                'region_id': row['region_id']
            } for _, row in ground_truth.iterrows()
        }

        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

        if not image_files:
            raise ValueError(f"Brak zdjęć w folderze {INPUT_DIR}")

        director = load_director()

        for image_file in image_files:
            image_path = os.path.join(INPUT_DIR, image_file)
            print(f"\n=== Przetwarzanie: {image_file} ===")
            try:
                if image_file not in ground_truth_dict:
                    print(f"Brak rzeczywistych danych dla {image_file} w pliku CSV")
                    continue
                true_data = ground_truth_dict[image_file]
                true_lat = true_data['latitude']
                true_lon = true_data['longitude']
                true_superregion = true_data['superRegion']
                true_region_id = true_data['region_id']

                inp = preprocess_image(image_path).to(DEVICE)

                with torch.no_grad():
                    pred_dir = director(inp)
                    region_id = pred_dir.argmax(dim=1).item()
                region_name = REGION_MAP.get(region_id, "Nieznany")
                print(f"Przewidywany region: {region_id} ({region_name})")
                print(f"Rzeczywisty superregion: {true_region_id} ({true_superregion})")

                expert = load_expert(region_id)
                with torch.no_grad():
                    pred_exp = expert(inp).cpu().numpy()[0]
                pred_lat, pred_lon = denormalize(pred_exp, REGION_PARAMS[region_id])
                maps_link = generate_google_maps_link(pred_lat, pred_lon)

                distance = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)

                print(f"Szerokość (latitude): {pred_lat:.6f} (rzeczywista: {true_lat:.6f})")
                print(f"Długość (longitude): {pred_lon:.6f} (rzeczywista: {true_lon:.6f})")
                print(f"Odległość od rzeczywistej lokalizacji: {distance:.2f} km")
                print(f"Link do Google Maps: {maps_link}")

            except Exception as e:
                print(f"Błąd dla obrazu {image_file}: {e}")

    except Exception as e:
        print(f"Błąd ogólny: {e}")