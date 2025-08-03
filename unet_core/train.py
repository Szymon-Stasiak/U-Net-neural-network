import os

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_core.model import UNet
import shutil
from unet_core.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

def train_fn(loader, model, optimizer, loss_fn, scaler, device="cuda" if torch.cuda.is_available() else "cpu"):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def train_process(
        train_img_dir="../data/shaped/images",
        train_mask_dir="../data/shaped/masks",
        val_img_dir="../data/shaped/images",
        val_mask_dir="../data/shaped/masks",
        learning_rate=1e-4,
        img_height=256,
        img_width=1918,
        batch_size=16,
        num_epochs=30,
        num_workers=4,
        pin_memory=True,
        load_model=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_predictions=False,
):
    train_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        val_img_dir=val_img_dir,
        val_mask_dir=val_mask_dir,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if load_model:
        load_checkpoint(torch.load("GlandFinder.pth.tar"), model, optimizer)

    best_dice_score = 0
    scaler = torch.amp.GradScaler(device if device == "cuda" else "cpu")
    checkpoint_path = "my_checkpoint.pth.tar"

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device)

        dice_score = check_accuracy(val_loader, model, device=device)

        if dice_score > best_dice_score:
            print(f"New best Dice score: {dice_score:.4f}. Saving checkpoint.")
            best_dice_score = dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            save_checkpoint(checkpoint)
        else:
            print("No improvement in Dice score. Skipping checkpoint save.")

        if (epoch + 1) % 50 == 0 and os.path.exists(checkpoint_path):
            backup_path = f"my_checkpoint_epoch_{epoch + 1}.pth.tar"
            shutil.copy(checkpoint_path, backup_path)
            print(f"Checkpoint copied to {backup_path}")

        if save_predictions:
            save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=device)

        model.train()
    return model
