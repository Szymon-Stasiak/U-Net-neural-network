import unittest
from unet_core.model import UNet
from unet_core.model import DoubleConv
import torch
import torch.nn.functional as F


class TestDoubleConv(unittest.TestCase):

    def test_forward(self):
        batch_size = 1
        in_channels = 3
        out_channels = 64
        height, width = 32, 32

        input_tensor = torch.randn(batch_size, in_channels, height, width)

        model = DoubleConv(in_channels, out_channels)

        output = model(input_tensor)

        self.assertEqual(output.shape, (batch_size, out_channels, height, width))

    def test_parameter_count(self):
        in_channels = 3
        out_channels = 64
        model = DoubleConv(in_channels, out_channels)

        num_params = sum(p.numel() for p in model.parameters())

        conv1_weight_params = in_channels * out_channels * 3 * 3
        conv2_weight_params = out_channels * out_channels * 3 * 3
        bn1_params = 2 * out_channels
        bn2_params = 2 * out_channels

        expected_params = conv1_weight_params + conv2_weight_params + bn1_params + bn2_params

        self.assertEqual(num_params, expected_params)

    def test_inference(self):
        batch_size = 1
        in_channels = 3
        out_channels = 64
        input_tensor = torch.randn(batch_size, in_channels, 32, 32)
        model = DoubleConv(in_channels, out_channels)

        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        self.assertEqual(output.shape, (batch_size, out_channels, 32, 32))


class TestUNet(unittest.TestCase):

    def test_forward(self):
        batch_size = 1
        in_channels = 3
        out_channels = 1
        height, width = 128, 128

        input_tensor = torch.randn(batch_size, in_channels, height, width)

        model = UNet(in_channels, out_channels)

        output = model(input_tensor)

        self.assertEqual(output.shape, (batch_size, out_channels, height, width))

    def test_parameter_count(self):
        in_channels = 3
        out_channels = 1
        model = UNet(in_channels, out_channels)

        num_params = sum(p.numel() for p in model.parameters())

        self.assertGreater(num_params, 0)

    def test_inference(self):
        batch_size = 1
        in_channels = 3
        out_channels = 1
        input_tensor = torch.randn(batch_size, in_channels, 128, 128)
        model = UNet(in_channels, out_channels)

        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        self.assertEqual(output.shape, (batch_size, out_channels, 128, 128))

    def test_skip_connections(self):
        batch_size = 1
        in_channels = 3
        out_channels = 1
        input_tensor = torch.randn(batch_size, in_channels, 128, 128)

        model = UNet(in_channels, out_channels)

        skip_connections = []
        x = input_tensor
        for encoder in model.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = model.pool(x)

        x = model.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(model.decoder), 2):
            x = model.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = model.decoder[idx + 1](concat_skip)

        output = model.final_conv(x)

        self.assertEqual(output.shape, (batch_size, out_channels, 128, 128))
