import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np



def prepare_image(image_path, img_width, img_height, device):
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((img_width, img_height))
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
    return image, tensor

def get_binary_mask(preds):
    return preds.float().squeeze().cpu().numpy().astype(np.uint8) * 255

def get_contours_from_mask(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


