import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
import multiprocessing
from tqdm import tqdm

DATA_DIR = '../../data/'
IMAGES_DIR = '../../data/raw/images'
METADATA_CSV = os.path.join(DATA_DIR, 'metadata_final.csv')
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_SIZE = (320, 240)
NUM_CLASSES = 13
MODEL_SAVE_PATH = '../../saved_models/director/efficientnet_b0_director_v2.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALID_RATIO = 0.1
SEED = 42

def validate_csv(csv_path, img_dir, num_classes):
    """
    Validates the CSV metadata and checks if image files exist.

    Args:
        csv_path (str): Path to the CSV file.
        img_dir (str): Directory containing image files.
        num_classes (int): Number of region classes.
    """
    df = pd.read_csv(csv_path)
    errors = []
    if 'filename' not in df.columns or 'region_id' not in df.columns:
        raise ValueError("CSV must contain columns 'filename' and 'region_id'")
    if df['filename'].isnull().any():
        errors.append("Missing filename in some rows")
    if df['region_id'].isnull().any():
        errors.append("Missing region_id in some rows")
    invalid_labels = df[~df['region_id'].between(0, num_classes - 1)]
    if not invalid_labels.empty:
        errors.append(f"Invalid region_id outside allowed range: \n{invalid_labels[['filename', 'region_id']].head()}")
    missing_files = [f for f in tqdm(df['filename'], desc="Validating CSV files", unit="file") if not os.path.isfile(os.path.join(img_dir, f))]
    if missing_files:
        errors.append(f"Missing files: {missing_files[:5]}... (total {len(missing_files)})")
    if errors:
        print("=== WARNINGS / ERRORS IN CSV ===")
        for e in errors:
            print(" -", e)
        print("=== Continuing despite warnings ===")
    else:
        print("CSV and files look correct.")

class StreetViewDataset(Dataset):
    """
    Custom Dataset for loading street view images and labels from a CSV file.
    """
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.images_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"[WARNING] Could not load {img_name}: {e}. Returning black image.")
            image = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))
        label = int(row['region_id'])
        if self.transform:
            image = self.transform(image)
        return image, label

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run computation on.

    Returns:
        avg_loss (float): Average loss for the epoch.
        accuracy (float): Accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(loader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            avg_loss = running_loss / total if total > 0 else 0
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}"})
    avg_loss = running_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def eval_epoch(model, loader, criterion, device):
    """
    Evaluates the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run computation on.

    Returns:
        avg_loss (float): Average validation loss.
        accuracy (float): Validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(loader, desc="Validation", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                avg_loss = running_loss / total if total > 0 else 0
                accuracy = correct / total if total > 0 else 0
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}"})
    avg_loss = running_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def main():
    """
    Main function to run the training and validation pipeline.
    """
    torch.manual_seed(SEED)
    validate_csv(METADATA_CSV, IMAGES_DIR, NUM_CLASSES)

    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset and splitting into train/valid...")
    full_dataset = StreetViewDataset(csv_file=METADATA_CSV, images_dir=IMAGES_DIR, transform=None)
    total_size = len(full_dataset)
    valid_size = int(total_size * VALID_RATIO)
    train_size = total_size - valid_size

    with tqdm(total=total_size, desc="Splitting dataset", unit="sample") as pbar:
        train_subset, valid_subset = random_split(full_dataset, [train_size, valid_size],
                                                  generator=torch.Generator().manual_seed(SEED))
        pbar.update(total_size)

    train_subset.dataset.transform = train_transforms
    valid_subset.dataset.transform = valid_transforms

    print("Preparing DataLoaders...")
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train samples: {len(train_subset)} | Valid samples: {len(valid_subset)} | Classes: {NUM_CLASSES}")
    print(f"Device: {DEVICE}")

    print("Loading EfficientNet...")
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=NUM_CLASSES)
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = eval_epoch(model, valid_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()