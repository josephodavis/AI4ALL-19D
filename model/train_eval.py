import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit  # <-- Added for advanced splitting

from model import FirstCNN
from dataset import BlindnessDataset

def train_one_epoch(model, loader, optimizer, criterion, device): 

    # train mode
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # progress bar with train dataloader
    loop = tqdm(loader, desc="  Train", leave=False)
    for images, labels in loop:
        # images + labels from dataloader
        images, labels = images.to(device), labels.to(device)

        # reset gradients
        optimizer.zero_grad()
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs, labels)
        # backpropagate
        loss.backward()
        optimizer.step()

        # add losses, correct preds, total preds, to calculate accuracy
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

    # return loss, accuracy
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    # evaluation mode
    model.eval()

    # keep track of loss, preds, and labels for accuracy + classification report
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # turn off gradient opimization
    with torch.no_grad():
        # validation progress bar
        loop = tqdm(loader, desc="  Val  ", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            loop.set_postfix(loss=f"{loss.item():.4f}")

    total = len(all_labels)
    # how many times prediction == label
    correct = sum(p == l for p, l in zip(all_preds, all_labels))

    # print classification report
    print(classification_report(
        all_labels, all_preds,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        zero_division=0
    ))
    
    return total_loss / total, correct / total

# Resolve paths from the project root regardless of where the notebook runs.
project_root = Path.cwd()
if project_root.name == "data":
    project_root = project_root.parent


def train(
    model,
    # Defaulting paths to match your new blended directory layout
    csv_path=project_root / "data" / "raw" / "2019_2015_data" / "traintestLabels15_trainLabels19.csv",
    image_dir=project_root / "data" / "raw" / "2019_2015_data" / "resized_traintest15_train19",
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
    val_split=0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Data augmentation pipeline for training data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

    # val pipeline no augmentation, just resizing and normalization
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    # Load master dataframe
    df = pd.read_csv(csv_path)
    
    # -------------------------------------------------------------
    # ADVANCED SPLITTING & DOWNSAMPLING LOGIC (Replaces random_split)
    # -------------------------------------------------------------
    np.random.seed(42)
    df['is_2015'] = df['image'].apply(lambda x: '_' in str(x))
    df_2015 = df[df['is_2015']].copy()
    df_2019 = df[~df['is_2015']].copy()

    # Step A: Stratified Split for 2019 Data (Perfect Class Balance)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    t_idx_19, v_idx_19 = next(sss.split(df_2019, df_2019['level']))
    train_2019 = df_2019.iloc[t_idx_19].copy()
    val_2019 = df_2019.iloc[v_idx_19].copy()

    # Step B: Patient-Group Split for 2015 Data (Anti-Data Leakage)
    df_2015['patient_id'] = df_2015['image'].apply(lambda x: str(x).split('_')[0])
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    t_idx_15, v_idx_15 = next(gss.split(df_2015, groups=df_2015['patient_id']))
    train_2015_raw = df_2015.iloc[t_idx_15].copy()
    val_2015 = df_2015.iloc[v_idx_15].copy()

    # Step C: Downsample 2015 Class 0 to Speed Up Training and Save Time
    train_2015_rare = train_2015_raw[train_2015_raw['level'] != 0]
    train_2015_zero = train_2015_raw[train_2015_raw['level'] == 0]
    
    # Cap 2015 Class 0 size to prevent overloading your hardware
    target_zero_count = train_2015_rare['level'].value_counts().max()
    zero_patients = train_2015_zero['patient_id'].unique()
    sampled_zero_patients = np.random.choice(
        zero_patients, 
        size=min(len(zero_patients), target_zero_count // 2), 
        replace=False
    )
    train_2015_zero_downsampled = train_2015_zero[train_2015_zero['patient_id'].isin(sampled_zero_patients)]
    train_2015_balanced = pd.concat([train_2015_rare, train_2015_zero_downsampled])

    # Step D: Combine into final clean DataFrames
    train_df = pd.concat([train_2019, train_2015_balanced], ignore_index=True).sample(frac=1, random_state=42)
    val_df = pd.concat([val_2019, val_2015], ignore_index=True).sample(frac=1, random_state=42)

    print(f"Engineered Training Set: {len(train_df)} images (Downsampled for efficiency)")
    print(f"Engineered Validation Set: {len(val_df)} images (Preserved for realistic testing)\n")

    # STEP E: WEIGHTED RANDOM SAMPLER LOGIC
    # -------------------------------------------------------------
    # 1. Get an array of all training labels
    train_labels = train_df['level'].values

    # 2. Count the occurrences of each class (0 through 4) in the training set
    class_counts = np.bincount(train_labels)

    # 3. Calculate the weight for each class (fewer items = higher weight)
    class_weights = 1.0 / class_counts

    # 4. Assign the corresponding class weight to every single sample row
    sample_weights = class_weights[train_labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    # 5. Initialize the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    # -------------------------------------------------------------

    # Wrap DataFrames into Datasets using the updated column headers
    train_set = BlindnessDataset(train_df, image_dir, transform=train_transform, id_col="image", label_col="level")
    val_set = BlindnessDataset(val_df, image_dir, transform=val_transform, id_col="image", label_col="level")

    # Construct DataLoaders (added pin_memory=True for faster VRAM delivery)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Track the absolute lowest validation loss across epochs
    best_val_loss = float('inf')

    # progress bar for specific epoch
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        # calculate training and validation loss, accuracy
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # print loss + accuracy
        epoch_bar.write(
            f"Epoch {epoch:>2}/{num_epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.3f}"
        )
        epoch_bar.set_postfix(
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.3f}",
        )

        # Check if this epoch's validation loss is lower than our previous best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the optimal configuration weights separately
            torch.save(model.state_dict(), "best_model.pth")
            epoch_bar.write(f"--> Found better weights! Saved checkpoint to best_model.pth (Val Loss: {best_val_loss:.4f})")

    # save and return trained model
    torch.save(model.state_dict(), "model.pth")
    print("Final epoch model saved to model.pth")
    print(f"Training absolute complete. Best validation loss achieved: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    train(FirstCNN())

# if __name__ == "__main__":
#     # load the pre-trained ResNet18 model
#     resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
#     # replace the final fully connected layer to match the number of classes (5 in this case)
#     in_features = resnet_model.fc.in_features
#     resnet_model.fc = nn.Linear(in_features, 5)
    
#     # train the modified ResNet18 model
#     train(resnet_model, num_epochs=10, lr=1e-4) 