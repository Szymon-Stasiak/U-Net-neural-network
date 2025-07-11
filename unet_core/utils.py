import torch
import torchvision
from unet_core.dataset.learn_set import LearnSet
from torch.utils.data import DataLoader
import os


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.eval()


def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size=16, train_transform=None,
                val_transform=None, num_workers=4, pin_memory=True):
    train_dataset = LearnSet(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )
    val_dataset = LearnSet(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_loader, val_loader


def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    acc = num_correct / num_pixels * 100
    avg_dice = dice_score / len(loader)
    print(f"Got {num_correct}/{num_pixels} with acc {acc:.2f}")
    print(f"Dice score: {avg_dice:.4f}")
    model.train()
    return avg_dice


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cpu"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
