import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

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
    ))
    
    return total_loss / total, correct / total



def train(
    model,
    csv_path="data/raw/aptos2019-blindness-detection/train.csv",
    image_dir="data/raw/aptos2019-blindness-detection/train_images",
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
    val_split=0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # "blueprint" to prepare images for input to model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # train.csv to pandas dataframe, then to torch dataset
    df = pd.read_csv(csv_path)
    dataset = BlindnessDataset(df, image_dir, transform=transform)

    # how many images in validation and train splits, then randomly assign samples to either
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # wrap datsets in dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = model.to(device)
    # use adam optimizer and CE loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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

    # save and return trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")
    return model

if __name__ == "__main__":
    train(FirstCNN())