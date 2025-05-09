
from unet_interface import UNET
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':

    # model = UNET.model(in_channels=3, out_channels=1)
    #
    # # Training
    # UNET.modelTrain(num_epochs=4)

    model = UNET._model(name="test_model")
    outlined_image = UNET.find("C:/Users/stszy/PycharmProjects/U-Net-neural-network/data/shaped/images/tile_12544_53760_7.tif")
    image_np = np.array(outlined_image)
    cv2.imshow("Mask", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    outlined_image = UNET.find_edges("C:/Users/stszy/PycharmProjects/U-Net-neural-network/data/shaped/images/tile_12544_53760_7.tif", color=(255, 0, 255))
    image_np = np.array(outlined_image )
    cv2.imshow("Mask", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    outlined_image = UNET.find_points("C:/Users/stszy/PycharmProjects/U-Net-neural-network/data/shaped/images/tile_12544_53760_7.tif")
    print(outlined_image)

