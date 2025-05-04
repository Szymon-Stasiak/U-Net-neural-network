import torch
import torchvision.transforms.functional as TF
import numpy as np
from model import UNet as UNetArchitecture
from train import train_process
from utils import load_checkpoint
import torch.optim as optim
from PIL import Image
import os
import cv2


class UNET:
    _loaded_model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def model(in_channels=3, out_channels=1, features=None, name=None):
        model = UNetArchitecture(in_channels=in_channels, out_channels=out_channels, features=features)
        model = model.to(UNET._device)

        if name == "GlandsFinder":
            checkpoint = torch.load("my_checkpoint.pth.tar", map_location=UNET._device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Dummy optimizer
            load_checkpoint(checkpoint, model, optimizer)
            print("✅ Pretrained model 'GlandsFinder' loaded.")
            model.eval()

        UNET._loaded_model = model
        return model

    @staticmethod
    def modelTrain(**kwargs):
        print("🚀 Starting training with parameters:", kwargs)
        train_process(**kwargs)

    @staticmethod
    def find(image_path):
        assert UNET._loaded_model is not None, "❌ Error: No pretrained model loaded. Call UNET.model(name='...') first."
        assert os.path.exists(image_path), f"❌ Image file not found: {image_path}"

        image = Image.open(image_path).convert("RGB")
        image = image.resize((256, 256))

        image_tensor = TF.to_tensor(image).unsqueeze(0).to(UNET._device)

        with torch.no_grad():
            preds = torch.sigmoid(UNET._loaded_model(image_tensor))
            preds = (preds > 0.5).float()

        mask = preds.squeeze().cpu().numpy()
        return mask

    @staticmethod
    def find_edges(image_path, color=(255, 0, 0)):

        assert UNET._loaded_model is not None, "❌ Error: No pretrained model loaded. Call UNET.model(name='...') first."
        assert os.path.exists(image_path), f"❌ Image file not found: {image_path}"

        original_image = Image.open(image_path).convert("RGB")
        resized_image = original_image.resize((256, 256))
        image_tensor = TF.to_tensor(resized_image).unsqueeze(0).to(UNET._device)

        with torch.no_grad():
            preds = torch.sigmoid(UNET._loaded_model(image_tensor))
            preds = (preds > 0.5).float()

        mask = preds.squeeze().cpu().numpy().astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = original_image.width / 256
        scale_y = original_image.height / 256
        scaled_contours = [np.array([[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)] for pt in cnt]).reshape(-1, 1, 2)
                           for cnt in contours]

        image_cv = np.array(original_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        cv2.drawContours(image_cv, scaled_contours, -1, color, thickness=2)

        result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        return result_image
    @staticmethod
    def find_points(image_path):

        assert UNET._loaded_model is not None, "❌ Error: No pretrained model loaded. Call UNET.model(name='...') first."
        assert os.path.exists(image_path), f"❌ Image file not found: {image_path}"

        original_image = Image.open(image_path).convert("RGB")
        resized_image = original_image.resize((256, 256))
        image_tensor = TF.to_tensor(resized_image).unsqueeze(0).to(UNET._device)

        with torch.no_grad():
            preds = torch.sigmoid(UNET._loaded_model(image_tensor))
            preds = (preds > 0.5).float()

        mask = preds.squeeze().cpu().numpy().astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = original_image.width / 256
        scale_y = original_image.height / 256

        all_points = []
        for cnt in contours:
            points = [(int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)) for pt in cnt]
            all_points.append(points)

        return all_points
