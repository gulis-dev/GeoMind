import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from math import radians, sin, cos, sqrt, atan2

DATA_DIR = '../../data/'
IMAGES_DIR = '../../data/raw/images'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata_final.csv')
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_SIZE = (320, 240)
DEVICE = torch.device('cuda')
MODEL_SAVE_DIR = '../../saved_models/experts/'
VAL_FRAC = 0.10
TEST_FRAC = 0.10

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points on the Earth specified in decimal degrees."""
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

class StreetViewRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, images_dir, transform=None, region_id=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        self.data = pd.read_csv(csv_file)
        if region_id is not None:
            self.data = self.data[self.data['region_id'] == region_id].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.lat_min = lat_min if lat_min is not None else self.data['latitude'].min()
        self.lat_max = lat_max if lat_max is not None else self.data['latitude'].max()
        self.lon_min = lon_min if lon_min is not None else self.data['longitude'].min()
        self.lon_max = lon_max if lon_max is not None else self.data['longitude'].max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.images_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))
        lat = (float(row['latitude']) - self.lat_min) / (self.lat_max - self.lat_min + 1e-8)
        lon = (float(row['longitude']) - self.lon_min) / (self.lon_max - self.lon_min + 1e-8)
        target = torch.tensor([lat, lon], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, target

def custom_haversine_loss(y_pred, y_true, lat_min, lat_max, lon_min, lon_max):
    """Custom loss function based on Haversine distance."""
    lat_pred = y_pred[:, 0] * (lat_max - lat_min) + lat_min
    lon_pred = y_pred[:, 1] * (lon_max - lon_min) + lon_min
    lat_true = y_true[:, 0] * (lat_max - lat_min) + lat_min
    lon_true = y_true[:, 1] * (lon_max - lon_min) + lon_min

    lat_pred_rad = torch.deg2rad(lat_pred)
    lon_pred_rad = torch.deg2rad(lon_pred)
    lat_true_rad = torch.deg2rad(lat_true)
    lon_true_rad = torch.deg2rad(lon_true)

    dlat = lat_pred_rad - lat_true_rad
    dlon = lon_pred_rad - lon_true_rad

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_true_rad) * torch.cos(lat_pred_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    R = 6371.0
    d = R * c
    return (d ** 2).mean()

def evaluate_distance(model, loader, lat_min, lat_max, lon_min, lon_max):
    """Evaluate the model's performance in terms of distance."""
    model.eval()
    distances = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            outputs = model(images).cpu().numpy()
            targets = targets.numpy()
            pred_lats = outputs[:, 0] * (lat_max - lat_min) + lat_min
            pred_lons = outputs[:, 1] * (lon_max - lon_min) + lon_min
            true_lats = targets[:, 0] * (lat_max - lat_min) + lat_min
            true_lons = targets[:, 1] * (lon_max - lon_min) + lon_min
            for i in range(len(outputs)):
                d = haversine_distance(pred_lats[i], pred_lons[i], true_lats[i], true_lons[i])
                distances.append(d)
    return np.mean(distances), np.median(distances), np.percentile(distances, 90)

def train_expert_regressor(region_id):
    """Train an expert regressor for a specific region."""
    print(f"\nTraining expert regressor for region {region_id}")
    df = pd.read_csv(METADATA_CSV)
    region_df = df[df['region_id'] == region_id]
    lat_min = region_df['latitude'].min()
    lat_max = region_df['latitude'].max()
    lon_min = region_df['longitude'].min()
    lon_max = region_df['longitude'].max()

    full_dataset = StreetViewRegressionDataset(
        csv_file=METADATA_CSV,
        images_dir=IMAGES_DIR,
        transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        region_id=region_id,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    N = len(full_dataset)
    if N == 0:
        print(f"No samples for region {region_id}")
        return

    val_size = int(N * VAL_FRAC)
    test_size = int(N * TEST_FRAC)
    train_size = N - val_size - test_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        print("Too few samples for splitting: train/val/test.")
        return
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, 2),
        nn.Sigmoid()
    )
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Region {region_id} Epoch {epoch+1}/{NUM_EPOCHS} [train]", unit="batch"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.clamp(outputs, 0, 1)
            loss = custom_haversine_loss(outputs, targets, lat_min, lat_max, lon_min, lon_max)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_train_loss = running_loss / train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                outputs = torch.clamp(outputs, 0, 1)
                loss = custom_haversine_loss(outputs, targets, lat_min, lat_max, lon_min, lon_max)
                val_loss += loss.item() * images.size(0)
        avg_val_loss = val_loss / val_size

        mean_dist, median_dist, p90_dist = evaluate_distance(model, val_loader, lat_min, lat_max, lon_min, lon_max)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | Val mean dist: {mean_dist:.2f}km | median: {median_dist:.2f}km | 90th perc: {p90_dist:.2f}km")

        scheduler.step(avg_val_loss)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, f"expert_regressor_{region_id}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved regressor for region {region_id} to {model_path}")


    mean_dist_train, median_dist_train, p90_dist_train = evaluate_distance(model, train_loader, lat_min, lat_max, lon_min, lon_max)
    mean_dist_val, median_dist_val, p90_dist_val = evaluate_distance(model, val_loader, lat_min, lat_max, lon_min, lon_max)
    mean_dist_test, median_dist_test, p90_dist_test = evaluate_distance(model, test_loader, lat_min, lat_max, lon_min, lon_max)
    print(f"Final TRAIN: mean {mean_dist_train:.2f}km, median {median_dist_train:.2f}km, 90th perc {p90_dist_train:.2f}km")
    print(f"Final VAL  : mean {mean_dist_val:.2f}km, median {median_dist_val:.2f}km, 90th perc {p90_dist_val:.2f}km")
    print(f"Final TEST : mean {mean_dist_test:.2f}km, median {median_dist_test:.2f}km, 90th perc {p90_dist_test:.2f}km")

def main():
    df = pd.read_csv(METADATA_CSV)
    region_ids = df['region_id'].unique()
    for region_id in region_ids:
        train_expert_regressor(region_id)

if __name__ == '__main__':
    main()