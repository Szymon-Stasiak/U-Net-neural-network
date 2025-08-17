from  unet_core.unet_interface import UNET

if __name__ == '__main__':
    unet = UNET(img_height=64, img_width=64)
    unet.set_model(in_channels=3, out_channels=1)

    unet.train_model(
        train_img_dir="../data/shaped/images",
        train_mask_dir="../data/shaped/masks",
        val_img_dir="../data/val/shaped/images",
        val_mask_dir="../data/val/shaped/masks",
        learning_rate=1e-4,
        batch_size=16,
        num_epochs=50,
        num_workers=4,
        pin_memory=True,
        save_predictions=True
    )
    print("UNET model is trained and ready to use. You can now use the model to make predictions or further train it.")
