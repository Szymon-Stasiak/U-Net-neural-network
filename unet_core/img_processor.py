import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from torchvision import transforms


def load_image_from_path(path):
    return Image.open(path).convert("RGB")


def prepare_image(image, img_width, img_height, device):
    original_image = image.copy()
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return original_image, image_tensor


def get_binary_mask(preds):
    return preds.float().squeeze().cpu().numpy().astype(np.uint8) * 255


def get_contours_from_mask(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
