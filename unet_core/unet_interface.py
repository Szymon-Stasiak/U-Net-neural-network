import os

import torch
import numpy as np
from unet_core.model import UNet as UNetArchitecture
from unet_core.train import train_process
from PIL import Image
import cv2
from unet_core.img_processor import prepare_image, get_binary_mask, get_contours_from_mask

class UNET:
    def __init__(self, device=None, img_height=256, img_width=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_height = img_height
        self.img_width = img_width
        self.model = None

    def set_model(self, in_channels=3, out_channels=1, features=None, name=None):
        self.model = UNetArchitecture(in_channels=in_channels, out_channels=out_channels, features=features).to(
            self.device)
        if name:
            self.model = self.model.load_pretrained_model(name=name, device=self.device)

    def train_model(self, **kwargs):
        kwargs['img_height'] = kwargs.get('img_height', self.img_height)
        kwargs['img_width'] = kwargs.get('img_width', self.img_width)
        kwargs['device'] = self.device
        print("Starting training with parameters:", kwargs)
        self.model = train_process(**kwargs)

    def find(self, image: Image.Image):
        original_image, preds = self.check_data(image)
        pred_mask = preds.squeeze().cpu().numpy().astype(np.uint8) * 255
        resized_mask = cv2.resize(
            pred_mask,
            (original_image.width, original_image.height),
            interpolation=cv2.INTER_NEAREST
        )
        return resized_mask

    def find_edges(self, image: Image.Image, color=(255, 0, 0)):
        contours, original_image, scale_x, scale_y = self.get_mask(image)

        scaled_contours = [
            np.array([[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)] for pt in cnt]).reshape(-1, 1, 2)
            for cnt in contours
        ]

        image_cv = np.array(original_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        cv2.drawContours(image_cv, scaled_contours, -1, color, thickness=2)

        result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)).resize(original_image.size)
        return result_image

    def find_points(self, image: Image.Image):
        contours, original_image, scale_x, scale_y = self.get_mask(image)

        all_points = []
        for cnt in contours:
            points = [(int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)) for pt in cnt]
            all_points.append(points)

        return all_points

    def get_mask(self, image: Image.Image):
        original_image, preds = self.check_data(image)
        mask = get_binary_mask(preds)
        contours = get_contours_from_mask(mask)
        scale_x = original_image.width / self.img_width
        scale_y = original_image.height / self.img_height
        return contours, original_image, scale_x, scale_y

    def check_data(self, image: Image.Image):
        assert self.model is not None, "Error: No pretrained model loaded. Call set_model() first or train your own."
        original_image, image_tensor = prepare_image(image, self.img_width, self.img_height, self.device)
        with torch.no_grad():
            preds = torch.sigmoid(self.model(image_tensor))
            preds = (preds > 0.5).float()
        return original_image, preds
