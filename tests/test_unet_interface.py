import pytest
from PIL import Image
from unet_core.unet_interface import UNET
import unittest


TEST_IMAGE_PATH = "file/test_img.tif"
MODEL_NAME = "file/test_model"
@pytest.fixture(scope="module", autouse=True)
def setup_model():
    UNET._model(name=MODEL_NAME)

class TestUNETInterface(unittest.TestCase):
    def test_find_output_size_matches_input(self):
        original = Image.open(TEST_IMAGE_PATH)
        mask = UNET.find(TEST_IMAGE_PATH)
        assert mask.shape == (original.height, original.width), \
            f"Expected output size {original.size[::-1]}, got {mask.shape}"

    def test_find_edges_output_size_matches_input(self):
        original = Image.open(TEST_IMAGE_PATH)
        result_img = UNET.find_edges(TEST_IMAGE_PATH)
        assert result_img.size == original.size, \
            f"Expected output size {original.size}, got {result_img.size}"

    def test_find_points_coordinates_within_bounds(self):
        original = Image.open(TEST_IMAGE_PATH)
        all_points = UNET.find_points(TEST_IMAGE_PATH)
        for contour in all_points:
            for x, y in contour:
                assert 0 <= x < original.width and 0 <= y < original.height, \
                    f"Point ({x}, {y}) out of bounds for image size {original.size}"
