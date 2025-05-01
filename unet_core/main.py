
from model import UNet
import torch

def print_hi(name):
    print(f'Hi, {name}')
    model = UNet(in_channels=1, out_channels=1)
    print(model)
    output = model(torch.randn(1, 1, 572, 572))


if __name__ == '__main__':
    print_hi('PyCharm')
    print_hi('PyCharm')
