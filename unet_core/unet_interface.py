import torch
import numpy as np
from unet_core.model import UNet as UNetArchitecture
from unet_core.train import train_process
from PIL import Image
import os
import cv2
from unet_core.imgProcessor import prepare_image, get_binary_mask, get_contours_from_mask


class UNET:
    _loaded_model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _img_height = 256
    _img_width = 256

    @staticmethod
    def _model(in_channels=3, out_channels=1, features=None, name=None, img_height=_img_height, img_width=_img_width):
        model = UNetArchitecture(in_channels=in_channels, out_channels=out_channels, features=features)
        model = model.to(UNET._device)

        if name:
            model = model.load_pretrained_model(name=name, device=UNET._device)
        UNET._img_height = img_height
        UNET._img_width = img_width
        UNET._loaded_model = model
        return model

    @staticmethod
    def _model_train(**kwargs):
        UNET._img_height = kwargs.get('img_height', UNET._img_height)
        UNET._img_width = kwargs.get('img_width', UNET._img_width)
        kwargs['img_height'] = UNET._img_height
        kwargs['img_width'] = UNET._img_width
        print("Starting training with parameters:", kwargs)
        train_process(**kwargs)

    @staticmethod
    def find(image_path):
        assert UNET._loaded_model is not None, "Error: No pretrained model loaded. Call UNET._model(name='...') first or train your own."
        assert os.path.exists(image_path), f"Image file not found: {image_path}"

        original_image, image_tensor = prepare_image(image_path, UNET._img_width, UNET._img_height, UNET._device)

        with torch.no_grad():
            preds = torch.sigmoid(UNET._loaded_model(image_tensor))
            preds = (preds > 0.5).float()
        pred_mask = preds.squeeze().cpu().numpy().astype(np.uint8) * 255  # shape: H x W
        resized_mask = cv2.resize(
            pred_mask,
            (original_image.width, original_image.height),
            interpolation=cv2.INTER_NEAREST
        )
        return resized_mask

    @staticmethod
    def find_edges(image_path, color=(255, 0, 0)):
        assert UNET._loaded_model is not None, "Error: No pretrained model loaded. Call UNET._model(name='...') first or train your own."
        assert os.path.exists(image_path), f"Image file not found: {image_path}"

        original_image, image_tensor = prepare_image(image_path, UNET._img_width, UNET._img_height, UNET._device)

        with torch.no_grad():
            preds = torch.sigmoid(UNET._loaded_model(image_tensor))
            preds = (preds > 0.5).float()

        mask = get_binary_mask(preds)
        contours = get_contours_from_mask(mask)

        scale_x = original_image.width / UNET._img_width
        scale_y = original_image.height / UNET._img_height

        scaled_contours = [
            np.array([[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)] for pt in cnt]).reshape(-1, 1, 2)
            for cnt in contours
        ]

        image_cv = np.array(original_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        cv2.drawContours(image_cv, scaled_contours, -1, color, thickness=2)

        result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)).resize(original_image.size)
        return result_image

    @staticmethod
    def find_points(image_path):
        assert UNET._loaded_model is not None, "Error: No pretrained model loaded. Call UNET._model(name='...') first or train your own."
        assert os.path.exists(image_path), f"Image file not found: {image_path}"

        original_image, image_tensor = prepare_image(image_path, UNET._img_width, UNET._img_height, UNET._device)

        with torch.no_grad():
            preds = torch.sigmoid(UNET._loaded_model(image_tensor))
            preds = (preds > 0.5).float()

        mask = get_binary_mask(preds)
        contours = get_contours_from_mask(mask)

        scale_x = original_image.width / UNET._img_width
        scale_y = original_image.height / UNET._img_height

        all_points = []
        for cnt in contours:
            points = [(int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)) for pt in cnt]
            all_points.append(points)

        return all_points
