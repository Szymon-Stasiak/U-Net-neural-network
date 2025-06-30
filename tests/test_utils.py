import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
from PIL import Image
from unet_core.utils import check_accuracy, save_predictions_as_imgs


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestSegmentationUtils(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.model = DummyModel().to(self.device)
        x = torch.randn(4, 1, 64, 64)
        y = torch.randint(0, 2, (4, 64, 64)).float()
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=2)

    def test_check_accuracy_output(self):
        dice_score = check_accuracy(self.loader, self.model, self.device)
        self.assertIsInstance(dice_score, torch.Tensor)
        self.assertTrue(0 <= dice_score.item() <= 1)

    def test_save_predictions_creates_files(self):
        folder = "test_images/"
        if os.path.exists(folder):
            shutil.rmtree(folder)

        save_predictions_as_imgs(self.loader, self.model, folder=folder, device=self.device)

        # Sprawdź, czy folder i pliki istnieją
        self.assertTrue(os.path.exists(folder))
        pred_files = [f for f in os.listdir(folder) if f.startswith("pred_")]
        label_files = [f for f in os.listdir(folder) if not f.startswith("pred_")]
        self.assertGreater(len(pred_files), 0)
        self.assertGreater(len(label_files), 0)

        # Sprawdź, czy zapisane obrazy są poprawne
        for file in pred_files + label_files:
            path = os.path.join(folder, file)
            with Image.open(path) as img:
                self.assertTrue(img.size[0] > 0 and img.size[1] > 0)

        shutil.rmtree(folder)


if __name__ == '__main__':
    unittest.main()
