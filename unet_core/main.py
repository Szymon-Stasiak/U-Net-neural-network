from unet_interface import UNET
import numpy as np
import cv2
from img_processor import load_image_from_path

if __name__ == '__main__':
    # unet = UNET(img_height=50, img_width=50)
    # unet.set_model(in_channels=3, out_channels=1)
    # unet.train_model(
    #     train_img_dir="../data/shaped/images",
    #     train_mask_dir="../data/shaped/masks",
    #     val_img_dir="../data/shaped/images",
    #     val_mask_dir="../data/shaped/masks",
    #     learning_rate=1e-4,
    #     batch_size=16,
    #     num_epochs=5,
    #     num_workers=4,
    #     pin_memory=True,
    #     save_predictions=True
    # )
    # outlined_image = unet.find(
    #     "C:/Users/stszy/PycharmProjects/U-Net-neural-network/data/shaped/images/tile_12544_53760_7.tif")
    # image_np = np.array(outlined_image)
    # cv2.imshow("Mask", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    unet = UNET(img_height=50, img_width=50)

    image = load_image_from_path(
        "C:/Users/stszy/PycharmProjects/U-Net-neural-network/data/shaped/images/tile_12544_53760_7.tif").convert("RGB")

    unet.set_model(in_channels=3, out_channels=1, name="my_checkpoint")

    outlined_image = unet.find(image)
    image_np = np.array(outlined_image)
    cv2.imshow("Mask", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    outlined_image = unet.find_edges(
        image,
        color=(255, 0, 255))
    image_np = np.array(outlined_image)
    cv2.imshow("Mask", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    outlined_image = unet.find_points(
        image)
    print(outlined_image)
