from unet_interface import UNET

if __name__ == '__main__':
    unet = UNET(img_height=512, img_width=512)
    unet.set_model(in_channels=3, out_channels=1)
    # unet.set_model(in_channels=3, out_channels=1, name="my_checkpoint")

    unet.train_model(
        train_img_dir="../data/shaped/images",
        train_mask_dir="../data/shaped/masks",
        val_img_dir="../data/val/shaped/images",
        val_mask_dir="../data/val/shaped/masks",
        learning_rate=1e-4,
        batch_size=16,
        num_epochs=250,
        num_workers=4,
        pin_memory=True,
        save_predictions=True
    )
    # outlined_image = unet.find(
    #     "C:/Users/stszy/PycharmProjects/U-Net-neural-network/data/shaped/images/tile_12544_53760_7.tif")
    # image_np = np.array(outlined_image)
    # cv2.imshow("Mask", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # unet = UNET(img_height=50, img_width=50)
    #
    print("UNET model is trained and ready to use. You can now use the model to make predictions or further train it.")
